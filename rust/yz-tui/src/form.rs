use std::fmt;

use crossterm::event::KeyModifiers;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    System,   // Inference server settings
    Search,   // MCTS algorithm params
    Pipeline, // Controller (iterations) + Self-play + Gating
    Learning, // Training + Model
    Data,     // Replay
}

impl Section {
    pub const ALL: [Section; 5] = [
        Section::System,
        Section::Search,
        Section::Pipeline,
        Section::Learning,
        Section::Data,
    ];

    pub fn title(&self) -> &'static str {
        match self {
            Section::System => "System",
            Section::Search => "Search",
            Section::Pipeline => "Pipeline",
            Section::Learning => "Learning",
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
    MctsTempCutoffPly,
    MctsVirtualLossMode,
    MctsVirtualLoss,

    // selfplay
    SelfplayGamesPerIteration,
    SelfplayWorkers,
    SelfplayThreadsPerWorker,

    // training
    TrainingMode, // Toggle between epochs and steps mode
    TrainingBatchSize,
    TrainingLearningRate,
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

    // replay
    ReplayCapacityShards,

    // controller
    ControllerTotalIterations,

    // model
    ModelHiddenDim,
    ModelNumBlocks,
    ModelKind,
}

impl FieldId {
    pub fn section(&self) -> Section {
        match self {
            // System: Inference server settings
            FieldId::InferBind
            | FieldId::InferDevice
            | FieldId::InferProtocolVersion
            | FieldId::InferLegalMaskBitset
            | FieldId::InferMaxBatch
            | FieldId::InferMaxWaitUs
            | FieldId::InferTorchThreads
            | FieldId::InferTorchInteropThreads
            | FieldId::InferDebugLog
            | FieldId::InferPrintStats => Section::System,

            // Search: MCTS algorithm params
            FieldId::MctsCPuct
            | FieldId::MctsBudgetReroll
            | FieldId::MctsBudgetMark
            | FieldId::MctsMaxInflightPerGame
            | FieldId::MctsDirichletAlpha
            | FieldId::MctsDirichletEpsilon
            | FieldId::MctsTempKind
            | FieldId::MctsTempT0
            | FieldId::MctsTempT1
            | FieldId::MctsTempCutoffPly
            | FieldId::MctsVirtualLossMode
            | FieldId::MctsVirtualLoss => Section::Search,

            // Pipeline: Controller (iterations) + Self-play + Gating
            FieldId::ControllerTotalIterations
            | FieldId::SelfplayGamesPerIteration
            | FieldId::SelfplayWorkers
            | FieldId::SelfplayThreadsPerWorker
            | FieldId::GatingGames
            | FieldId::GatingSeed
            | FieldId::GatingSeedSetId
            | FieldId::GatingWinRateThreshold
            | FieldId::GatingPairedSeedSwap
            | FieldId::GatingDeterministicChance => Section::Pipeline,

            // Learning: Training + Model
            FieldId::TrainingMode
            | FieldId::TrainingBatchSize
            | FieldId::TrainingLearningRate
            | FieldId::TrainingEpochs
            | FieldId::TrainingWeightDecay
            | FieldId::TrainingStepsPerIteration
            | FieldId::ModelHiddenDim
            | FieldId::ModelNumBlocks
            | FieldId::ModelKind => Section::Learning,

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
            FieldId::MctsTempCutoffPly => "mcts.temperature_schedule.cutoff_ply",
            FieldId::MctsVirtualLossMode => "mcts.virtual_loss_mode",
            FieldId::MctsVirtualLoss => "mcts.virtual_loss",

            FieldId::SelfplayGamesPerIteration => "selfplay.games_per_iteration",
            FieldId::SelfplayWorkers => "selfplay.workers",
            FieldId::SelfplayThreadsPerWorker => "selfplay.threads_per_worker",

            FieldId::TrainingMode => "training.mode",
            FieldId::TrainingBatchSize => "training.batch_size",
            FieldId::TrainingLearningRate => "training.learning_rate",
            FieldId::TrainingEpochs => "training.epochs",
            FieldId::TrainingWeightDecay => "training.weight_decay",
            FieldId::TrainingStepsPerIteration => "training.steps_per_iteration",

            FieldId::GatingGames => "gating.games",
            FieldId::GatingSeed => "gating.seed",
            FieldId::GatingSeedSetId => "gating.seed_set_id",
            FieldId::GatingWinRateThreshold => "gating.win_rate_threshold",
            FieldId::GatingPairedSeedSwap => "gating.paired_seed_swap",
            FieldId::GatingDeterministicChance => "gating.deterministic_chance",

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
    // System: Inference server settings
    FieldId::InferBind,
    FieldId::InferDevice,
    FieldId::InferProtocolVersion,
    FieldId::InferLegalMaskBitset,
    FieldId::InferMaxBatch,
    FieldId::InferMaxWaitUs,
    FieldId::InferTorchThreads,
    FieldId::InferTorchInteropThreads,
    FieldId::InferDebugLog,
    FieldId::InferPrintStats,
    // Search: MCTS algorithm params
    FieldId::MctsCPuct,
    FieldId::MctsBudgetReroll,
    FieldId::MctsBudgetMark,
    FieldId::MctsMaxInflightPerGame,
    FieldId::MctsDirichletAlpha,
    FieldId::MctsDirichletEpsilon,
    FieldId::MctsTempKind,
    FieldId::MctsTempT0,
    FieldId::MctsTempT1,
    FieldId::MctsTempCutoffPly,
    FieldId::MctsVirtualLossMode,
    FieldId::MctsVirtualLoss,
    // Pipeline: Controller (iterations) + Self-play + Gating
    FieldId::ControllerTotalIterations,
    FieldId::SelfplayGamesPerIteration,
    FieldId::SelfplayWorkers,
    FieldId::SelfplayThreadsPerWorker,
    FieldId::GatingGames,
    FieldId::GatingSeed,
    FieldId::GatingSeedSetId,
    FieldId::GatingWinRateThreshold,
    FieldId::GatingPairedSeedSwap,
    FieldId::GatingDeterministicChance,
    // Learning: Training + Model
    FieldId::TrainingMode,
    FieldId::TrainingBatchSize,
    FieldId::TrainingLearningRate,
    FieldId::TrainingEpochs,
    FieldId::TrainingWeightDecay,
    FieldId::TrainingStepsPerIteration,
    FieldId::ModelHiddenDim,
    FieldId::ModelNumBlocks,
    FieldId::ModelKind,
    // Data: Replay
    FieldId::ReplayCapacityShards,
];
