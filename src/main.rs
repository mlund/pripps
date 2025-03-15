// Copyright 2025 Isabel Vinterbladh, Sergei Grudinin and Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

use anyhow::Result;
use clap::Parser;
use pretty_env_logger::env_logger::DEFAULT_FILTER_ENV;
use pripps::{AtomKind, Structure, cli};
use std::io::Write;

use std::process::ExitCode;
extern crate pretty_env_logger;

// Wrapper for main function to handle errors
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
    let formfactors = match &args.atomfile {
        Some(atompath) => pripps::formfactor::read_formfactors(atompath)?,
        None => anyhow::bail!("missing formfactor file"),
    };
    let atomkinds = formfactors.keys().cloned().collect::<Vec<AtomKind>>();

    match args.command {
        cli::Commands::Debye {
            input,
            qmin,
            qmax,
            nq,
            output,
        } => {
            let structure = Structure::from_xyz(&input, &atomkinds)?;
            let intensity =
                pripps::debye::calc_intensity(&structure, &formfactors, (qmin, qmax), nq)?;
            let mut output = std::fs::File::create(output)?;
            writeln!(output, "# q/Å I(q)")?;
            for (q, i) in intensity {
                writeln!(output, "{:.6} {:.6}", q, i)?;
            }
        }
    };
    Ok(())
}

fn main() -> ExitCode {
    match do_main() {
        Err(err) => {
            eprintln!("error: {}", &err);
            ExitCode::FAILURE
        }
        Ok(_) => ExitCode::SUCCESS,
    }
}
