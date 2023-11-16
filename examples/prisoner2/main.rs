use std::collections::HashMap;
use std::path::PathBuf;
use std::thread;
use log::LevelFilter;
use amfi::agent::{AgentGenT, AutomaticAgentRewarded, ReinitAgent, StatefulAgent, TracingAgent};
use amfi::comm::SyncCommEnv;
use amfi::env::generic::HashMapEnvT;
use amfi::env::{ReinitEnvironment, RoundRobinUniversalEnvironment, TracingEnv};
use amfi::error::AmfiError;
use amfi_examples::classic::agent::{CoverPolicy, Forgive1Policy, PrisonerInfoSet, RandomPrisonerPolicy, SwitchOnTwoSubsequent};
use amfi_examples::classic::common::SymmetricRewardTableInt;
use amfi_examples::classic::domain::ClassicAction::Defect;
use amfi_examples::classic::domain::ClassicGameDomain;
use amfi_examples::classic::domain::PrisonerId::{Andrzej, Janusz};
use amfi_examples::classic::env::PrisonerEnvState;


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




fn main() -> Result<(), AmfiError<ClassicGameDomain>>{
    println!("Hello prisoners;");
    setup_logger(LevelFilter::Debug, &None).unwrap();

    let reward_table = SymmetricRewardTableInt::new(5, 1, 10, 3);
    /*{
        /*coop_when_coop: 5,
        defect_when_coop: 10,
        defect_when_defect: 3,
        coop_when_defect: 1*/
        //enum_map!()
    };*/


    let env_state = PrisonerEnvState::new(reward_table,  10);

    let (comm_env_0, comm_prisoner_0) = SyncCommEnv::new_pair();
    let (comm_env_1, comm_prisoner_1) = SyncCommEnv::new_pair();

    let mut prisoner0 = AgentGenT::new(
        Andrzej,
        PrisonerInfoSet::new(reward_table.clone()), comm_prisoner_0, CoverPolicy{});

    let mut prisoner1 = AgentGenT::new(
        Janusz,
        PrisonerInfoSet::new(reward_table.clone()), comm_prisoner_1, Forgive1Policy{});

    let mut env_coms = HashMap::new();
    env_coms.insert(Andrzej, comm_env_0);
    env_coms.insert(Janusz, comm_env_1);

    let mut env = HashMapEnvT::new(env_state, env_coms);

    thread::scope(|s|{
        s.spawn(||{
            env.run_round_robin_uni_rewards().unwrap();
        });
        s.spawn(||{
            prisoner0.run_rewarded().unwrap();
        });
        s.spawn(||{
            prisoner1.run_rewarded().unwrap();
        });
    });

    println!("Scenario 2");


    /*
    let env_state = PrisonerEnvState::new(reward_table,  100);

    let (comm_env_0, comm_prisoner_0) = SyncCommEnv::new_pair();
    let (comm_env_1, comm_prisoner_1) = SyncCommEnv::new_pair();

    let mut prisoner0 = AgentGenT::new(
        Andrzej,
        PrisonerState::new(reward_table), comm_prisoner_0, RandomPrisonerPolicy{});

    let mut prisoner1 = AgentGenT::new(
        Janusz,
        PrisonerState::new(reward_table), comm_prisoner_1, BetrayRatioPolicy{});

    let mut env_coms = HashMap::new();
    env_coms.insert(Andrzej, comm_env_0);
    env_coms.insert(Janusz, comm_env_1);

    let mut env = TracingGenericEnv::new( env_state, env_coms);

     */

    env.reinit(PrisonerEnvState::new(reward_table.clone(), 10));
    let mut prisoner0 = prisoner0.transform_replace_policy(RandomPrisonerPolicy{});
    //let mut prisoner1 = prisoner1.do_change_policy(BetrayRatioPolicy{});
    let mut prisoner1 = prisoner1.transform_replace_policy(SwitchOnTwoSubsequent{});
    prisoner0.reinit(PrisonerInfoSet::new(reward_table.clone()));
    prisoner1.reinit(PrisonerInfoSet::new(reward_table.clone()));

    thread::scope(|s|{
        s.spawn(||{
            env.run_round_robin_uni_rewards().unwrap();
        });
        s.spawn(||{
            prisoner0.run_rewarded().unwrap();
        });
        s.spawn(||{
            prisoner1.run_rewarded().unwrap();
        });
    });

    let prisoner0_betrayals = prisoner0.info_set().count_actions(Defect);
    let prisoner1_betrayals = prisoner1.info_set().count_actions(Defect);

    println!("Prisoner 0 betrayed {:?} times and Prisoner 1 betrayed {:?} times.", prisoner0_betrayals, prisoner1_betrayals);

    for elem in env.trajectory().list(){
        println!("{}", elem);
    }

    for trace in prisoner1.game_trajectory().list(){
        println!("{}", trace);
    }



    Ok(())
}