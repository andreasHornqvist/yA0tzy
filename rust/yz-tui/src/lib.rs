//! Ratatui terminal UI (TUI) for configuring and monitoring runs.
//!
//! v1 scope:
//! - basic screen routing + key handling
//! - run picker (list/create)

use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::{execute, ExecutableCommand};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use ratatui::Terminal;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    Home,
    Config,
    Dashboard,
}

#[derive(Debug)]
struct App {
    screen: Screen,
    status: String,
    runs_dir: PathBuf,
    runs: Vec<String>,
    selected: usize,

    active_run_id: Option<String>,
    cfg: yz_core::Config,

    metrics_tail: Vec<String>,
}

impl App {
    fn new(runs_dir: PathBuf) -> Self {
        Self {
            screen: Screen::Home,
            status: "q: quit | r: refresh | n: new run".to_string(),
            runs_dir,
            runs: Vec::new(),
            selected: 0,
            active_run_id: None,
            cfg: yz_core::Config::default(),
            metrics_tail: Vec::new(),
        }
    }

    fn refresh_runs(&mut self) {
        self.runs = list_runs(&self.runs_dir);
        if self.selected >= self.runs.len() {
            self.selected = 0;
        }
        if self.runs.is_empty() {
            self.status = format!(
                "No runs found. Press 'n' to create one in {}",
                self.runs_dir.display()
            );
        } else {
            self.status = "q: quit | r: refresh | n: new run | ↑/↓: select".to_string();
        }
    }

    fn create_run(&mut self) -> io::Result<()> {
        std::fs::create_dir_all(&self.runs_dir)?;
        let ts = yz_logging::now_ms();
        let id = format!("run_{ts}");
        let dir = self.runs_dir.join(&id);
        std::fs::create_dir_all(dir.join("logs"))?;
        std::fs::create_dir_all(dir.join("models"))?;
        std::fs::create_dir_all(dir.join("replay"))?;
        self.status = format!("Created {id}");
        self.refresh_runs();
        self.selected = self.runs.iter().position(|r| r == &id).unwrap_or(0);
        Ok(())
    }

    fn enter_selected_run(&mut self) {
        if self.runs.is_empty() {
            return;
        }
        self.active_run_id = Some(self.runs[self.selected].clone());
        self.cfg = yz_core::Config::default();
        self.screen = Screen::Config;
        self.status =
            "Config: ←/→ edit games_per_iteration | s: save config.yaml | Esc: back | q: quit"
                .to_string();
    }

    fn run_dir(&self) -> Option<PathBuf> {
        self.active_run_id
            .as_ref()
            .map(|id| self.runs_dir.join(id))
    }

    fn save_config_snapshot(&mut self) {
        let Some(run_dir) = self.run_dir() else {
            self.status = "No run selected".to_string();
            return;
        };
        if let Err(e) = std::fs::create_dir_all(&run_dir) {
            self.status = format!("save failed: {e}");
            return;
        }
        match yz_logging::write_config_snapshot_atomic(&run_dir, &self.cfg) {
            Ok((rel, h)) => self.status = format!("Saved {rel} (hash {h})"),
            Err(e) => self.status = format!("save failed: {e:?}"),
        }
    }

    fn refresh_metrics_tail(&mut self) {
        let Some(run_dir) = self.run_dir() else {
            self.metrics_tail.clear();
            return;
        };
        let metrics = run_dir.join("logs").join("metrics.ndjson");
        let mut out: Vec<String> = Vec::new();
        if let Ok(s) = std::fs::read_to_string(metrics) {
            let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
            let n = 20usize.min(lines.len());
            out.extend(lines[lines.len().saturating_sub(n)..].iter().map(|x| x.to_string()));
        }
        self.metrics_tail = out;
    }

    fn enter_dashboard(&mut self) {
        if self.active_run_id.is_none() {
            return;
        }
        self.refresh_metrics_tail();
        self.screen = Screen::Dashboard;
        self.status = "Dashboard: r refresh | Esc back | q quit".to_string();
    }
}

fn list_runs(runs_dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(runs_dir) else {
        return out;
    };
    for ent in rd.flatten() {
        let Ok(ft) = ent.file_type() else { continue };
        if !ft.is_dir() {
            continue;
        }
        let name = ent.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        out.push(name);
    }
    out.sort();
    out
}

pub fn run() -> io::Result<()> {
    // Terminal init.
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut app = App::new(PathBuf::from("runs"));
    app.refresh_runs();

    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| draw(f, &app))?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                match app.screen {
                    Screen::Home => match k.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char('r') => app.refresh_runs(),
                        KeyCode::Char('n') => {
                            if let Err(e) = app.create_run() {
                                app.status = format!("create failed: {e}");
                            }
                        }
                        KeyCode::Enter => app.enter_selected_run(),
                        KeyCode::Up => {
                            if app.selected > 0 {
                                app.selected -= 1;
                            }
                        }
                        KeyCode::Down => {
                            if app.selected + 1 < app.runs.len() {
                                app.selected += 1;
                            }
                        }
                        _ => {}
                    },
                    Screen::Config => match k.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Esc => {
                            app.screen = Screen::Home;
                            app.active_run_id = None;
                            app.refresh_runs();
                        }
                        KeyCode::Char('s') => app.save_config_snapshot(),
                        KeyCode::Char('d') => app.enter_dashboard(),
                        KeyCode::Left => {
                            app.cfg.selfplay.games_per_iteration =
                                app.cfg.selfplay.games_per_iteration.saturating_sub(1);
                        }
                        KeyCode::Right => {
                            app.cfg.selfplay.games_per_iteration =
                                app.cfg.selfplay.games_per_iteration.saturating_add(1);
                        }
                        _ => {}
                    },
                    Screen::Dashboard => match k.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Esc => {
                            app.screen = Screen::Config;
                            app.status = "Config: ←/→ edit games_per_iteration | s: save config.yaml | d: dashboard | Esc: back | q: quit".to_string();
                        }
                        KeyCode::Char('r') => app.refresh_metrics_tail(),
                        _ => {}
                    },
                }
            }
        }
        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    // Terminal restore.
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn draw(f: &mut ratatui::Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)].as_ref())
        .split(f.area());

    match app.screen {
        Screen::Home => {
            let title = Line::from(vec![
                Span::styled("yA0tzy", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::raw("TUI"),
            ]);
            let items: Vec<ListItem> = if app.runs.is_empty() {
                vec![ListItem::new(Line::from("No runs yet"))]
            } else {
                app.runs
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        let prefix = if i == app.selected { "> " } else { "  " };
                        ListItem::new(Line::from(format!("{prefix}{r}")))
                    })
                    .collect()
            };
            let list = List::new(items).block(Block::default().title(title).borders(Borders::ALL));
            f.render_widget(list, chunks[0]);
        }
        Screen::Config => {
            let rid = app.active_run_id.as_deref().unwrap_or("<no run>");
            let title = Line::from(vec![
                Span::styled("Config", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::raw(format!("run={rid}")),
            ]);
            let body = vec![
                Line::from(format!(
                    "selfplay.games_per_iteration = {}",
                    app.cfg.selfplay.games_per_iteration
                )),
                Line::from(format!("inference.bind = {}", app.cfg.inference.bind)),
                Line::from(format!("mcts.budget_mark = {}", app.cfg.mcts.budget_mark)),
                Line::from(""),
                Line::from("v1 editor: only games_per_iteration is editable here"),
                Line::from("press 'd' for dashboard (tails logs/metrics.ndjson)"),
            ];
            let p = Paragraph::new(body)
                .block(Block::default().title(title).borders(Borders::ALL));
            f.render_widget(p, chunks[0]);
        }
        Screen::Dashboard => {
            let rid = app.active_run_id.as_deref().unwrap_or("<no run>");
            let title = Line::from(vec![
                Span::styled("Dashboard", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::raw(format!("run={rid}")),
            ]);
            let mut body: Vec<Line> = Vec::new();
            body.push(Line::from("Tail: logs/metrics.ndjson (last 20 lines)"));
            body.push(Line::from(""));
            if app.metrics_tail.is_empty() {
                body.push(Line::from("(no metrics yet)"));
            } else {
                for l in &app.metrics_tail {
                    body.push(Line::from(l.clone()));
                }
            }
            let p = Paragraph::new(body)
                .block(Block::default().title(title).borders(Borders::ALL));
            f.render_widget(p, chunks[0]);
        }
    }

    let help = Paragraph::new(app.status.clone())
        .block(Block::default().title("Status").borders(Borders::ALL));
    f.render_widget(help, chunks[1]);
}


