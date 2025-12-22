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
                    println!("Oracle expected score: ~248.44 (not yet implemented)");
                }
                "sim" => {
                    println!("Oracle simulation (not yet implemented)");
                    println!("Usage: yz oracle sim --games N --seed S");
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

