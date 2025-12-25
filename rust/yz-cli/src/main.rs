//! yz: CLI binary for AlphaZero Yatzy.
//!
//! Subcommands (PRD section 13):
//! - oracle expected
//! - oracle sim
//! - selfplay
//! - gate
//! - oracle-eval
//! - bench
//! - profile

use std::env;
use std::process;

/// Print the oracle's optimal expected score for a fresh game.
fn cmd_oracle_expected() {
    println!("Building oracle DP table...");
    let info = yz_oracle::get_expected_score();
    println!();
    println!("Optimal expected score: {:.4}", info.expected_score);
    println!("Build time: {:.2}s", info.build_time_secs);
}

fn cmd_oracle_sim(args: &[String]) {
    let mut games: usize = 10_000;
    let mut seed: u64 = 0;
    let mut no_hist = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz oracle sim

USAGE:
    yz oracle sim [--games N] [--seed S] [--no-hist]

OPTIONS:
    --games N    Number of games to simulate (default: 10000)
    --seed S     RNG seed (default: 0)
    --no-hist    Skip printing histogram
"#
                );
                return;
            }
            "--games" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for --games");
                    process::exit(1);
                }
                games = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --games value: {}", args[i + 1]);
                    process::exit(1);
                });
                i += 2;
            }
            "--seed" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for --seed");
                    process::exit(1);
                }
                seed = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --seed value: {}", args[i + 1]);
                    process::exit(1);
                });
                i += 2;
            }
            "--no-hist" => {
                no_hist = true;
                i += 1;
            }
            other => {
                eprintln!("Unknown option for `yz oracle sim`: {}", other);
                eprintln!("Run `yz oracle sim --help` for usage.");
                process::exit(1);
            }
        }
    }

    println!("Building oracle DP table...");
    let _ = yz_oracle::oracle(); // build+cache
    println!("Running simulation...");

    let report = yz_oracle::simulate(games, seed);
    let s = report.summary;

    println!();
    println!("Evaluation:");
    println!("  - Games: {}", games);
    println!(
        "  - Score: mean={:.2}, median={}, std={:.2}, min={}, max={}",
        s.mean, s.median, s.std_dev, s.min, s.max
    );
    println!("  - Upper bonus rate: {:.1}%", report.bonus_rate * 100.0);

    if !no_hist {
        yz_oracle::print_histogram(&report.scores);
    }
}

fn print_help() {
    eprintln!(
        r#"yz - AlphaZero Yatzy CLI

USAGE:
    yz <COMMAND> [OPTIONS]

COMMANDS:
    oracle expected     Print oracle expected score (~248.44)
    oracle sim          Run oracle solitaire simulation
    selfplay            Run self-play with MCTS + inference
    gate                Gate candidate vs best model
    oracle-eval         Evaluate models against oracle baseline
    bench               Run micro-benchmarks
    profile             Run with profiler hooks enabled

OPTIONS:
    -h, --help          Print this help message
    -V, --version       Print version

For more information, see the PRD or run `yz <COMMAND> --help`.
"#
    );
}

fn print_version() {
    println!("yz {}", env!("CARGO_PKG_VERSION"));
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        process::exit(0);
    }

    match args[1].as_str() {
        "-h" | "--help" | "help" => {
            print_help();
        }
        "-V" | "--version" => {
            print_version();
        }
        "oracle" => {
            if args.len() < 3 {
                eprintln!("Usage: yz oracle <expected|sim> [OPTIONS]");
                process::exit(1);
            }
            match args[2].as_str() {
                "expected" => {
                    cmd_oracle_expected();
                }
                "sim" => {
                    cmd_oracle_sim(&args[3..]);
                }
                _ => {
                    eprintln!("Unknown oracle subcommand: {}", args[2]);
                    process::exit(1);
                }
            }
        }
        "selfplay" => {
            println!("Self-play (not yet implemented)");
            println!("Usage: yz selfplay --config cfg.yaml --infer unix:///... --out runs/<id>/");
        }
        "gate" => {
            println!("Gating (not yet implemented)");
            println!("Usage: yz gate --config cfg.yaml --best best.pt --cand cand.pt");
        }
        "oracle-eval" => {
            println!("Oracle evaluation (not yet implemented)");
            println!("Usage: yz oracle-eval --config cfg.yaml --best ... --cand ...");
        }
        "bench" => {
            println!("Benchmarks (not yet implemented)");
        }
        "profile" => {
            println!("Profiling (not yet implemented)");
            println!("Usage: yz profile selfplay ...");
        }
        cmd => {
            eprintln!("Unknown command: {}", cmd);
            eprintln!("Run `yz --help` for usage.");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cli_compiles() {
        // Basic sanity: the binary compiles and this test runs.
        assert!(true);
    }
}
