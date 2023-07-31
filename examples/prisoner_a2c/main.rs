use std::path::PathBuf;
use log::LevelFilter;
use sztorm::comm::SyncCommEnv;
use sztorm::error::SztormError;
use sztorm_examples::prisoner::common::RewardTable;
use sztorm_examples::prisoner::domain::PrisonerDomain;
use sztorm_examples::prisoner::env::PrisonerEnvState;
pub fn setup_logger(log_level: LevelFilter, log_file: &Option<PathBuf>) -> Result<(), fern::InitError> {
    let dispatch  = fern::Dispatch::new()

        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level(log_level);

        match log_file{
            None => dispatch.chain(std::io::stdout()),
            Some(f) => dispatch.chain(fern::log_file(f)?)
        }

        //.chain(std::io::stdout())
        //.chain(fern::log_file("output.log")?)
        .apply()?;
    Ok(())
}
fn main() -> Result<(), SztormError<PrisonerDomain>>{

    setup_logger(LevelFilter::Debug, &None).unwrap();

    let reward_table = RewardTable{
        cover_v_cover: 5,
        betray_v_cover: 10,
        betray_v_betray: 3,
        cover_v_betray: 1
    };


    let env_state = PrisonerEnvState::new(reward_table,  10);

    //let (comm_env_0, comm_prisoner_0) = SyncCommEnv::new_pair();
    //let (comm_env_1, comm_prisoner_1) = SyncCommEnv::new_pair();

    Ok(())
}