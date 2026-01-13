use std::fmt;

use crossterm::event::KeyModifiers;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Inference,
    Search,
    Pipeline,
    Model,
    Data,
}

impl Section {
    pub const ALL: [Section; 5] = [
        Section::Inference,
        Section::Search,
        Section::Pipeline,
        Section::Model,
        Section::Data,
    ];

    pub fn title(&self) -> &'static str {
        match self {
            Section::Inference => "Inference",
            Section::Search => "Search",
            Section::Pipeline => "Pipeline",
            Section::Model => "Model",
            Section::Data => "Data",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldId {
    // inference
    InferBind,
    InferDevice,
    InferProtocolVersion,
    InferLegalMaskBitset,
    InferMaxBatch,
    InferMaxWaitUs,
    InferTorchThreads,
    InferTorchInteropThreads,
    InferDebugLog,
    InferPrintStats,

    // mcts
    MctsCPuct,
    MctsBudgetReroll,
    MctsBudgetMark,
    MctsMaxInflightPerGame,
    MctsDirichletAlpha,
    MctsDirichletEpsilon,
    MctsTempKind,
    MctsTempT0,
    MctsTempT1,
    MctsTempCutoffTurn,
    MctsVirtualLossMode,
    MctsVirtualLoss,
    MctsKatagoExpansionLock,

    // selfplay
    SelfplayGamesPerIteration,
    SelfplayWorkers,
    SelfplayThreadsPerWorker,

    // training
    TrainingMode, // Toggle between epochs and steps mode
    TrainingBatchSize,
    TrainingLearningRate,
    TrainingContinuousCandidateTraining,
    TrainingResetOptimizer,
    TrainingOptimizer,
    TrainingEpochs,
    TrainingWeightDecay,
    TrainingStepsPerIteration,

    // gating
    GatingGames,
    GatingSeed,
    GatingSeedSetId,
    GatingWinRateThreshold,
    GatingPairedSeedSwap,
    GatingDeterministicChance,
    GatingKatagoSprt,
    GatingKatagoSprtMinGames,
    GatingKatagoSprtMaxGames,
    GatingKatagoSprtAlpha,
    GatingKatagoSprtBeta,
    GatingKatagoSprtDelta,

    // replay
    ReplayCapacityShards,

    // controller
    ControllerTotalIterations,

    // model
    ModelHiddenDim,
    ModelNumBlocks,
    ModelKind,
}

#[allow(dead_code)]
const _HIDDEN_INFERENCE_FIELDS: [FieldId; 2] = [FieldId::InferDebugLog, FieldId::InferPrintStats];

impl FieldId {
    pub fn section(&self) -> Section {
        match self {
            // Inference
            FieldId::InferBind
            | FieldId::InferDevice
            | FieldId::InferProtocolVersion
            | FieldId::InferLegalMaskBitset
            | FieldId::InferMaxBatch
            | FieldId::InferMaxWaitUs
            | FieldId::InferTorchThreads
            | FieldId::InferTorchInteropThreads
            | FieldId::InferDebugLog
            | FieldId::InferPrintStats => Section::Inference,

            // Search: MCTS
            | FieldId::MctsCPuct
            | FieldId::MctsBudgetReroll
            | FieldId::MctsBudgetMark
            | FieldId::MctsMaxInflightPerGame
            | FieldId::MctsDirichletAlpha
            | FieldId::MctsDirichletEpsilon
            | FieldId::MctsTempKind
            | FieldId::MctsTempT0
            | FieldId::MctsTempT1
            | FieldId::MctsTempCutoffTurn
            | FieldId::MctsVirtualLossMode
            | FieldId::MctsVirtualLoss
            | FieldId::MctsKatagoExpansionLock => Section::Search,

            // Pipeline: Controller + Self-play + Training + Gating
            FieldId::ControllerTotalIterations
            | FieldId::SelfplayGamesPerIteration
            | FieldId::SelfplayWorkers
            | FieldId::SelfplayThreadsPerWorker
            | FieldId::TrainingMode
            | FieldId::TrainingBatchSize
            | FieldId::TrainingLearningRate
            | FieldId::TrainingContinuousCandidateTraining
            | FieldId::TrainingResetOptimizer
            | FieldId::TrainingOptimizer
            | FieldId::TrainingEpochs
            | FieldId::TrainingWeightDecay
            | FieldId::TrainingStepsPerIteration
            | FieldId::GatingGames
            | FieldId::GatingSeed
            | FieldId::GatingSeedSetId
            | FieldId::GatingWinRateThreshold
            | FieldId::GatingPairedSeedSwap
            | FieldId::GatingDeterministicChance
            | FieldId::GatingKatagoSprt
            | FieldId::GatingKatagoSprtMinGames
            | FieldId::GatingKatagoSprtMaxGames
            | FieldId::GatingKatagoSprtAlpha
            | FieldId::GatingKatagoSprtBeta
            | FieldId::GatingKatagoSprtDelta => Section::Pipeline,

            // Model
            FieldId::ModelHiddenDim
            | FieldId::ModelNumBlocks
            | FieldId::ModelKind => Section::Model,

            // Data: Replay
            FieldId::ReplayCapacityShards => Section::Data,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            FieldId::InferBind => "inference.bind",
            FieldId::InferDevice => "inference.device",
            FieldId::InferProtocolVersion => "inference.protocol_version",
            FieldId::InferLegalMaskBitset => "inference.legal_mask_bitset",
            FieldId::InferMaxBatch => "inference.max_batch",
            FieldId::InferMaxWaitUs => "inference.max_wait_us",
            FieldId::InferTorchThreads => "inference.torch_threads",
            FieldId::InferTorchInteropThreads => "inference.torch_interop_threads",
            FieldId::InferDebugLog => "inference.debug_log (YZ_DEBUG_LOG)",
            FieldId::InferPrintStats => "inference.print_stats (YZ_INFER_PRINT_STATS)",

            FieldId::MctsCPuct => "mcts.c_puct",
            FieldId::MctsBudgetReroll => "mcts.budget_reroll",
            FieldId::MctsBudgetMark => "mcts.budget_mark",
            FieldId::MctsMaxInflightPerGame => "mcts.max_inflight_per_game",
            FieldId::MctsDirichletAlpha => "mcts.dirichlet_alpha",
            FieldId::MctsDirichletEpsilon => "mcts.dirichlet_epsilon",
            FieldId::MctsTempKind => "mcts.temperature_schedule.kind",
            FieldId::MctsTempT0 => "mcts.temperature_schedule.t0",
            FieldId::MctsTempT1 => "mcts.temperature_schedule.t1",
            FieldId::MctsTempCutoffTurn => "mcts.temperature_schedule.cutoff_turn",
            FieldId::MctsVirtualLossMode => "mcts.virtual_loss_mode",
            FieldId::MctsVirtualLoss => "mcts.virtual_loss",
            FieldId::MctsKatagoExpansionLock => "mcts.katago.expansion_lock",

            FieldId::SelfplayGamesPerIteration => "selfplay.games_per_iteration",
            FieldId::SelfplayWorkers => "selfplay.workers",
            FieldId::SelfplayThreadsPerWorker => "selfplay.threads_per_worker",

            FieldId::TrainingMode => "training.mode",
            FieldId::TrainingBatchSize => "training.batch_size",
            FieldId::TrainingLearningRate => "training.learning_rate",
            FieldId::TrainingContinuousCandidateTraining => "training.continuous_candidate_training (Space to toggle)",
            FieldId::TrainingResetOptimizer => "training.reset_optimizer (Space to toggle)",
            FieldId::TrainingOptimizer => "training.optimizer",
            FieldId::TrainingEpochs => "training.epochs",
            FieldId::TrainingWeightDecay => "training.weight_decay",
            FieldId::TrainingStepsPerIteration => "training.steps_per_iteration",

            FieldId::GatingGames => "gating.games",
            FieldId::GatingSeed => "gating.seed",
            FieldId::GatingSeedSetId => "gating.seed_set_id",
            FieldId::GatingWinRateThreshold => "gating.win_rate_threshold",
            FieldId::GatingPairedSeedSwap => "gating.paired_seed_swap",
            FieldId::GatingDeterministicChance => "gating.deterministic_chance",
            FieldId::GatingKatagoSprt => "gating.katago.sprt",
            FieldId::GatingKatagoSprtMinGames => "gating.katago.sprt_min_games",
            FieldId::GatingKatagoSprtMaxGames => "gating.katago.sprt_max_games",
            FieldId::GatingKatagoSprtAlpha => "gating.katago.sprt_alpha (false promote rate α)",
            FieldId::GatingKatagoSprtBeta => "gating.katago.sprt_beta (false reject rate β)",
            FieldId::GatingKatagoSprtDelta => "gating.katago.sprt_delta (± band around threshold)",

            FieldId::ReplayCapacityShards => "replay.capacity_shards",

            FieldId::ControllerTotalIterations => "controller.total_iterations",

            FieldId::ModelHiddenDim => "model.hidden_dim",
            FieldId::ModelNumBlocks => "model.num_blocks",
            FieldId::ModelKind => "model.kind",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditMode {
    View,
    Editing,
}

#[derive(Debug, Clone)]
pub struct FormState {
    pub selected_idx: usize,
    pub edit_mode: EditMode,
    pub input_buf: String,
    pub last_validation_error: Option<String>,
}

impl Default for FormState {
    fn default() -> Self {
        Self {
            selected_idx: 0,
            edit_mode: EditMode::View,
            input_buf: String::new(),
            last_validation_error: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepSize {
    Small,
    Large,
}

impl StepSize {
    pub fn from_mods(m: KeyModifiers) -> Self {
        if m.contains(KeyModifiers::SHIFT) {
            StepSize::Large
        } else {
            StepSize::Small
        }
    }
}

impl fmt::Display for StepSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepSize::Small => write!(f, "small"),
            StepSize::Large => write!(f, "large"),
        }
    }
}

pub const ALL_FIELDS: &[FieldId] = &[
    // Inference
    FieldId::InferBind,
    FieldId::InferDevice,
    FieldId::InferProtocolVersion,
    FieldId::InferLegalMaskBitset,
    FieldId::InferMaxBatch,
    FieldId::InferMaxWaitUs,
    FieldId::InferTorchThreads,
    FieldId::InferTorchInteropThreads,
    // (intentionally hidden from TUI: inference.debug_log, inference.print_stats)

    // Search
    FieldId::MctsCPuct,
    FieldId::MctsBudgetReroll,
    FieldId::MctsBudgetMark,
    FieldId::MctsMaxInflightPerGame,
    FieldId::MctsDirichletAlpha,
    FieldId::MctsDirichletEpsilon,
    FieldId::MctsTempKind,
    FieldId::MctsTempT0,
    FieldId::MctsTempT1,
    FieldId::MctsTempCutoffTurn,
    FieldId::MctsVirtualLossMode,
    FieldId::MctsVirtualLoss,
    FieldId::MctsKatagoExpansionLock,

    // Pipeline
    FieldId::ControllerTotalIterations,
    FieldId::SelfplayGamesPerIteration,
    FieldId::SelfplayWorkers,
    FieldId::SelfplayThreadsPerWorker,
    FieldId::TrainingMode,
    FieldId::TrainingBatchSize,
    FieldId::TrainingLearningRate,
    FieldId::TrainingContinuousCandidateTraining,
    FieldId::TrainingResetOptimizer,
    FieldId::TrainingOptimizer,
    FieldId::TrainingEpochs,
    FieldId::TrainingWeightDecay,
    FieldId::TrainingStepsPerIteration,
    FieldId::GatingKatagoSprt,
    FieldId::GatingGames,
    FieldId::GatingKatagoSprtMinGames,
    FieldId::GatingKatagoSprtMaxGames,
    FieldId::GatingKatagoSprtAlpha,
    FieldId::GatingKatagoSprtBeta,
    FieldId::GatingKatagoSprtDelta,
    FieldId::GatingSeedSetId,
    FieldId::GatingSeed,
    FieldId::GatingPairedSeedSwap,
    FieldId::GatingDeterministicChance,
    FieldId::GatingWinRateThreshold,

    // Model
    FieldId::ModelHiddenDim,
    FieldId::ModelNumBlocks,
    FieldId::ModelKind,

    // Data
    FieldId::ReplayCapacityShards,
];
