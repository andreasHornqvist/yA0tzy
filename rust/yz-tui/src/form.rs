use std::fmt;

use crossterm::event::KeyModifiers;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Inference,
    Mcts,
    Selfplay,
    Training,
    Gating,
    Replay,
    Controller,
}

impl Section {
    pub const ALL: [Section; 7] = [
        Section::Inference,
        Section::Mcts,
        Section::Selfplay,
        Section::Training,
        Section::Gating,
        Section::Replay,
        Section::Controller,
    ];

    pub fn title(&self) -> &'static str {
        match self {
            Section::Inference => "Inference",
            Section::Mcts => "MCTS",
            Section::Selfplay => "Self-play",
            Section::Training => "Training",
            Section::Gating => "Gating",
            Section::Replay => "Replay",
            Section::Controller => "Controller",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldId {
    // inference
    InferBind,
    InferDevice,
    InferMaxBatch,
    InferMaxWaitUs,

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

    // selfplay
    SelfplayGamesPerIteration,
    SelfplayWorkers,
    SelfplayThreadsPerWorker,

    // training
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
}

impl FieldId {
    pub fn section(&self) -> Section {
        match self {
            FieldId::InferBind
            | FieldId::InferDevice
            | FieldId::InferMaxBatch
            | FieldId::InferMaxWaitUs => Section::Inference,

            FieldId::MctsCPuct
            | FieldId::MctsBudgetReroll
            | FieldId::MctsBudgetMark
            | FieldId::MctsMaxInflightPerGame
            | FieldId::MctsDirichletAlpha
            | FieldId::MctsDirichletEpsilon
            | FieldId::MctsTempKind
            | FieldId::MctsTempT0
            | FieldId::MctsTempT1
            | FieldId::MctsTempCutoffPly => Section::Mcts,

            FieldId::SelfplayGamesPerIteration
            | FieldId::SelfplayWorkers
            | FieldId::SelfplayThreadsPerWorker => Section::Selfplay,

            FieldId::TrainingBatchSize | FieldId::TrainingLearningRate | FieldId::TrainingEpochs => {
                Section::Training
            }
            FieldId::TrainingWeightDecay | FieldId::TrainingStepsPerIteration => Section::Training,

            FieldId::GatingGames
            | FieldId::GatingSeed
            | FieldId::GatingSeedSetId
            | FieldId::GatingWinRateThreshold
            | FieldId::GatingPairedSeedSwap
            | FieldId::GatingDeterministicChance => Section::Gating,

            FieldId::ReplayCapacityShards => Section::Replay,

            FieldId::ControllerTotalIterations => Section::Controller,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            FieldId::InferBind => "inference.bind",
            FieldId::InferDevice => "inference.device",
            FieldId::InferMaxBatch => "inference.max_batch",
            FieldId::InferMaxWaitUs => "inference.max_wait_us",

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

            FieldId::SelfplayGamesPerIteration => "selfplay.games_per_iteration",
            FieldId::SelfplayWorkers => "selfplay.workers",
            FieldId::SelfplayThreadsPerWorker => "selfplay.threads_per_worker",

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
    // inference
    FieldId::InferBind,
    FieldId::InferDevice,
    FieldId::InferMaxBatch,
    FieldId::InferMaxWaitUs,
    // mcts
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
    // selfplay
    FieldId::SelfplayGamesPerIteration,
    FieldId::SelfplayWorkers,
    FieldId::SelfplayThreadsPerWorker,
    // training
    FieldId::TrainingBatchSize,
    FieldId::TrainingLearningRate,
    FieldId::TrainingEpochs,
    FieldId::TrainingWeightDecay,
    FieldId::TrainingStepsPerIteration,
    // gating
    FieldId::GatingGames,
    FieldId::GatingSeed,
    FieldId::GatingSeedSetId,
    FieldId::GatingWinRateThreshold,
    FieldId::GatingPairedSeedSwap,
    FieldId::GatingDeterministicChance,
    // replay
    FieldId::ReplayCapacityShards,
    // controller
    FieldId::ControllerTotalIterations,
];


