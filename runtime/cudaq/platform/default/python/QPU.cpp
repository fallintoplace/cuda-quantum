/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QPU.h"
#include "common/ArgumentWrapper.h"
#include "common/CompiledModule.h"
#include "common/Environment.h"
#include "common/ExecutionContext.h"
#include "common/RuntimeTarget.h"
#include "common/Timing.h"
#include "cudaq_internal/compiler/ArgumentConversion.h"
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/JIT.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "cudaq_internal/compiler/TracePassInstrumentation.h"
#include "nvqir/resourcecounter/ResourceCounterScope.h"
#include "runtime/cudaq/platform/PythonSignalCheck.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/AddMetadata.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Optimizer/Transforms/ResourceCount.h"
#include "cudaq/Target/CompileTarget.h"
#include "cudaq/Verifier/QIRLLVMIRDialect.h"
#include "cudaq/platform.h"
#include "cudaq/runtime/logger/logger.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include <cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h>

using namespace mlir;

static void specializeKernel(const std::string &name, ModuleOp module,
                             std::span<void *const> rawArgs, Type resultTy = {},
                             bool enablePythonCodegenDump = false,
                             bool isEntryPoint = true,
                             bool isFullySpecialized = true) {
  PassManager pm(module.getContext());
  cudaq::addPythonSignalInstrumentation(pm);
  pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
  cudaq_internal::compiler::ArgumentConverter argCon(name, module);
  // Look up the kernel's type signature.
  argCon.gen(rawArgs);

  SmallVector<std::string> kernels;
  SmallVector<std::string> substs;
  for (auto *kInfo : argCon.getKernelSubstitutions()) {
    std::string kernName =
        cudaq::runtime::cudaqGenPrefixName + kInfo->getKernelName().str();
    kernels.emplace_back(kernName);
    std::string substBuff;
    llvm::raw_string_ostream ss(substBuff);
    ss << kInfo->getSubstitutionModule();
    substs.emplace_back(substBuff);
  }

  // Collect references for the argument synthesis.
  llvm::SmallVector<llvm::StringRef> kernelRefs{kernels.begin(), kernels.end()};
  llvm::SmallVector<llvm::StringRef> substRefs{substs.begin(), substs.end()};

  // Run a pass manager to specialize & optimize the kernel to be launched.
  pm.addPass(cudaq::opt::createArgumentSynthesisPass(
      kernelRefs, substRefs, /*changeSemantics=*/false));
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(cudaq::opt::createLambdaLifting({.constantPropagation = true}));
  // We must inline these lambda calls before apply specialization as it does
  // not perform control/adjoint specialization across function call boundary.
  cudaq::opt::addAggressiveInlining(pm);
  pm.addPass(
      cudaq::opt::createApplySpecialization({.constantPropagation = true}));
  cudaq::opt::addAggressiveInlining(pm);
  pm.addPass(cudaq::opt::createDistributedDeviceCall());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // Run GKE to generate `.thunk` / `.argsCreator` when the kernel has a result
  // or any unspecialized arguments so they can be properly marshaled
  if (isEntryPoint && (resultTy || !isFullySpecialized)) {
    pm.addPass(cudaq::opt::createGenerateKernelExecution(
        {.positNullary = isFullySpecialized, .ignoreHostFunction = true}));
    pm.addPass(cudaq::opt::createRunSemanticsHackery());
  }
  pm.addPass(mlir::createSymbolDCEPass());
  if (enablePythonCodegenDump) {
    module.getContext()->disableMultithreading();
    pm.enableIRPrinting();
  }
  if (failed(cudaq::runPassManagerReleasingGIL(pm, module)))
    throw std::runtime_error("Pass pipeline failed.");
}

/// Run the target compilation pipeline for Python MLIR kernels. Returns true
/// when the target pipeline already ran standard finalization.
static bool runTargetPassPipeline(mlir::ModuleOp module) {
  auto *rt = cudaq::get_platform().get_runtime_target();
  if (!rt)
    return false;
  cudaq::CompileTarget ct(rt->config, rt->runtimeConfig,
                          cudaq::is_emulated_platform());
  const auto &pipelineConfig = ct.pipelineConfig;
  auto passPipeline = cudaq_internal::compiler::getPassPipeline(ct);

  auto *ctx = module.getContext();
  PassManager pm(ctx);
  cudaq::addPythonSignalInstrumentation(pm);
  pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
  std::string errMsg;
  llvm::raw_string_ostream errOS(errMsg);
  if (mlir::failed(mlir::parsePassPipeline(passPipeline, pm, errOS)))
    throw std::runtime_error("Failed to parse target pipeline: " + errMsg);
  if (failed(cudaq::runPassManagerReleasingGIL(pm, module)))
    throw std::runtime_error("Pass pipeline failed.");
  return pipelineConfig.runsStandardFinalize;
}

/// Lowers \p module to LLVM code. The LLVM code will use "full QIR" as the
/// transport layer. If \p kernelName and \p args are provided, they will
/// specialize the selected entry-point kernel.
std::string cudaq::detail::lower_to_qir_llvm(const std::string &name,
                                             mlir::ModuleOp module,
                                             OpaqueArguments &args,
                                             const std::string &format) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "getQIR", name);
  // Translate the module to QIR transport layer (as LLVM code).
  cudaq_internal::compiler::mergeAllCallableClosures(module, name,
                                                     args.getArgs());
  specializeKernel(name, module, args.getArgs());
  const bool finalizedByTargetPipeline = runTargetPassPipeline(module);
  PassManager pm(module.getContext());
  cudaq::addPythonSignalInstrumentation(pm);
  pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
  if (!finalizedByTargetPipeline) {
    cudaq::opt::addAggressiveInlining(pm);
    cudaq::opt::createTargetFinalizePipeline(pm);
  }
  cudaq::opt::addAOTPipelineConvertToQIR(pm, format);
  if (failed(cudaq::runPassManagerReleasingGIL(pm, module)))
    throw std::runtime_error("Pass pipeline failed.");
  if (failed(cudaq::verifier::checkQIRLLVMIRDialect(module, format)))
    throw std::runtime_error("QIR conformance failed.");
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule)
    return "{translation failed}";
  std::string result;
  llvm::raw_string_ostream os(result);
  llvmModule->print(os, nullptr);
  os.flush();
  return result;
}

/// Lowers \p module to `Open QASM 2`. The output will be a string of `Open
/// QASM` code. \p kernelName and \p args should be provided, as they will
/// specialize the selected entry-point kernel.
std::string cudaq::detail::lower_to_openqasm(const std::string &name,
                                             mlir::ModuleOp module,
                                             OpaqueArguments &args) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "getASM", name);
  // Translate module to OpenQASM2 transport layer.
  cudaq_internal::compiler::mergeAllCallableClosures(module, name,
                                                     args.getArgs());
  specializeKernel(name, module, args.getArgs());
  const bool finalizedByTargetPipeline = runTargetPassPipeline(module);
  auto *ctx = module.getContext();
  PassManager pm(ctx);
  cudaq::addPythonSignalInstrumentation(pm);
  pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
  if (!finalizedByTargetPipeline)
    cudaq::opt::createTargetFinalizePipeline(pm);
  cudaq::opt::createPipelineTransformsForPythonToOpenQASM(pm);
  cudaq::opt::addPipelineTranslateToOpenQASM(pm);
  const bool enablePrintMLIRBeforeAndAfterEachPass =
      cudaq::getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", false);
  if (enablePrintMLIRBeforeAndAfterEachPass) {
    ctx->disableMultithreading();
    pm.enableIRPrinting();
  }
  if (failed(cudaq::runPassManagerReleasingGIL(pm, module)))
    throw std::runtime_error("Pass pipeline failed.");
  std::string result;
  llvm::raw_string_ostream os(result);
  if (mlir::failed(cudaq::translateToOpenQASM(module, os)))
    return "{translation failed}";
  os.flush();
  return result;
}

static std::optional<cudaq::JitEngine>
alreadyBuiltJITCode(const std::string &name) {
  auto *currentExecCtx = cudaq::getExecutionContext();
  if (currentExecCtx && currentExecCtx->allowJitEngineCaching) {
    auto jit = currentExecCtx->jitEng;
    if (jit && cudaq::compiler_artifact::isPersistingJITEngine()) {
      CUDAQ_INFO("Loading previously compiled JIT engine for {}. This will "
                 "re-run the previous job, discarding any changes to the "
                 "kernel, arguments or launch configuration.",
                 currentExecCtx->kernelName);
      cudaq::compiler_artifact::checkArtifactReuse(name, jit.value());
    }
    return jit;
  }

  // Fallback for callers without an ExecutionContext (e.g. direct kernel
  // calls): look up the artifact saved by a previous compilation.
  return cudaq::compiler_artifact::getArtifactJit(name);
}

/// In a sample launch context, the (`JIT` compiled) execution engine may be
/// cached so that it can be called many times in a loop without being
/// recompiled. This exploits the fact that the arguments processed at the
/// sample callsite are invariant by the definition of a `CUDA-Q` kernel.
static void cacheJITForPerformance(cudaq::JitEngine jit) {
  auto *currentExecCtx = cudaq::getExecutionContext();
  if (currentExecCtx && currentExecCtx->allowJitEngineCaching) {
    if (!currentExecCtx->jitEng)
      currentExecCtx->jitEng = jit;
  }
}

std::unique_ptr<cudaq::CompileTarget>
cudaq::ModuleLauncher::getCompileTarget() {
  return getCompileTarget(cudaq::getExecutionContext());
}

namespace {
struct PythonLauncher : public cudaq::ModuleLauncher {
  class PythonLauncherCompileTarget : public cudaq::CompileTarget {
    using cudaq::CompileTarget::CompileTarget;
    void addPassInstrumentation(mlir::PassManager &pm) const override {
      cudaq::addPythonSignalInstrumentation(pm);
      pm.addInstrumentation(
          std::make_unique<cudaq::TracePassInstrumentation>());
    }
    void withGilReleased(const std::function<void()> &fn) const override {
      cudaq::withGilReleased(fn);
    }
  };

  using cudaq::ModuleLauncher::getCompileTarget;
  std::unique_ptr<cudaq::CompileTarget>
  getCompileTarget(cudaq::ExecutionContext *context) override {
    const bool enablePythonCodegenDump =
        cudaq::getEnvBool("CUDAQ_PYTHON_CODEGEN_DUMP", false);
    if (enablePythonCodegenDump) {
      CUDAQ_WARN("CUDAQ_PYTHON_CODEGEN_DUMP is no longer supported and will be "
                 "ignored. Use CUDAQ_MLIR_PRINT_EACH_PASS instead.");
    }
    std::unique_ptr<cudaq::CompileTarget> ct;
    auto *rt = cudaq::get_platform().get_runtime_target();
    if (!rt) {
      ct = std::make_unique<PythonLauncherCompileTarget>();
      ct->pipelineConfig.runsStandardFinalize = false;
    } else {
      ct = std::make_unique<PythonLauncherCompileTarget>(
          rt->config, rt->runtimeConfig, cudaq::is_emulated_platform());
    }

    bool isLocalSimulator =
        !(cudaq::is_remote_platform() || cudaq::is_emulated_platform());

    ct->fullySpecialize = !isLocalSimulator;
    ct->pipelineConfig.addDistributedDeviceCall = true;
    ct->generateResourceCounts = context && context->name == "resource-count";
    ct->argumentSynthChangeSemantics = false;
    ct->pipelineConfig.codegenTranslation = "qir:";
    ct->emitJit = true;
    return ct;
  }

  cudaq::CompiledModule compileModule(const cudaq::SourceModule &src,
                                      cudaq::KernelArgs args,
                                      bool isEntryPoint) override {

    ScopedTraceWithContext(cudaq::TIMING_LAUNCH,
                           "PythonLauncher::compileModule");
    const auto &kernelName = src.getName();
    auto modulePtr = src.getMlirOpaqueModulePtr();
    assert(modulePtr &&
           "PythonLauncher::compileModule requires an MLIR artifact");

    cudaq_internal::compiler::Compiler compiler(getCompileTarget());

    ModuleOp module = ModuleOp::getFromOpaquePointer(modulePtr);

    std::string fullName = cudaq::runtime::cudaqGenPrefixName + kernelName;

    auto funcOp = module.lookupSymbol<mlir::func::FuncOp>(fullName);
    if (!funcOp)
      throw std::runtime_error("no kernel named " + kernelName +
                               " found in module");
    mlir::Type resultTy = cudaq::runtime::getReturnType(funcOp);

    auto resultInfo =
        cudaq_internal::compiler::CompiledModuleHelper::createResultInfo(
            resultTy, isEntryPoint, module);

    // Determine whether the kernel needs argument packing (argsCreator) by
    // checking if any non-callable arguments are present. This must be done
    // before the cache lookup so the cached path uses the correct value.
    bool isFullySpecialized = true;

    { // JIT caching -- TODO: isFullySpecialized is WRONG here
      if (auto jit = alreadyBuiltJITCode(kernelName)) {
        auto jitArtifacts =
            cudaq_internal::compiler::CompiledModuleHelper::createJitArtifacts(
                kernelName, *jit, resultInfo, isFullySpecialized);
        return cudaq_internal::compiler::CompiledModuleHelper::
            createCompiledModule(kernelName, resultInfo, jitArtifacts);
      }
    }

    auto compiled =
        compiler.runPassPipeline(kernelName, modulePtr, args, isEntryPoint);
    auto jit = compiled.getJit().value();

    cacheJITForPerformance(jit.getEngine());
    cudaq::compiler_artifact::saveArtifact(kernelName, jit.getEngine());

    return compiled;
  }
};
} // namespace

// PythonLauncher registration. This TU only builds into the Python extension
// (_quakeDialects.so), but `launchModule` / `specializeModule` live in
// libcudaq.so. CUDA-Q Registry uses `static inline Head/Tail`, so each DSO
// that instantiates the template gets its own copy — `CUDAQ_REGISTER_TYPE`
// would add the node to the extension's (unseen-by-libcudaq) registry. We
// instead call the `cudaq_add_module_launcher_node` bridge defined in
// libcudaq.so so the registration lands in the registry that `launchModule`
// actually reads. Mirrors the `cudaq_add_qpu_node` pattern used for QPUs.
extern "C" void cudaq_add_module_launcher_node(void *node_ptr);

namespace {
struct PythonLauncherRegistration {
  cudaq::RegistryEntry<cudaq::ModuleLauncher> entry;
  cudaq::Registry<cudaq::ModuleLauncher>::node node;
  PythonLauncherRegistration()
      : entry("default", &PythonLauncherRegistration::ctorFn), node(entry) {
    cudaq_add_module_launcher_node(&node);
  }
  static std::unique_ptr<cudaq::ModuleLauncher> ctorFn() {
    return std::make_unique<PythonLauncher>();
  }
};
static PythonLauncherRegistration s_pythonLauncherRegistration;
} // namespace

extern "C" void cudaq_ensure_default_launcher_linked(void) {}
