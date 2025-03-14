use anyhow::Result;
use clap::Parser;
use pretty_env_logger::env_logger::DEFAULT_FILTER_ENV;
use pripps::{AtomKind, Structure, cli};

use std::process::ExitCode;
extern crate pretty_env_logger;

// Wrapper for main function to handle errors
#[allow(unused)]
fn do_main() -> Result<()> {
    let args = cli::Args::parse();
    // Configure logging
    if args.verbose {
        unsafe { std::env::set_var(DEFAULT_FILTER_ENV, "Debug") };
    } else if std::env::var(DEFAULT_FILTER_ENV).is_err() {
        unsafe { std::env::set_var(DEFAULT_FILTER_ENV, "Info") };
    }
    pretty_env_logger::init();

    // load form factor models
    let formfactors = match &args.atoms {
        Some(atompath) => pripps::read_formfactors(&atompath)?,
        None => anyhow::bail!("No atom definition file provided"),
    };
    let atomkinds = formfactors.keys().cloned().collect::<Vec<AtomKind>>();

    match args.command {
        cli::Commands::Debye {
            input,
            qmin,
            qmax,
            nq,
        } => {
            let structure = Structure::from_xyz(&input, &atomkinds)?;
        }
    };
    Ok(())
}

fn main() -> ExitCode {
    if let Err(err) = do_main() {
        eprintln!("Error: {}", &err);
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
