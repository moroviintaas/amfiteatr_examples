use std::collections::HashMap;
use std::path::PathBuf;
use std::thread;
use log::LevelFilter;
use sztorm::agent::{AgentGenT, AutomaticAgentRewarded, ResetAgent, StatefulAgent, TracingAgent};
use sztorm::comm::SyncCommEnv;
use sztorm::env::generic::HashMapEnvT;
use sztorm::env::{ResetEnvironment,  RoundRobinUniversalEnvironment, TracingEnv};
use sztorm::error::SztormError;
use sztorm_examples::prisoner::agent::{CoverPolicy, Forgive1Policy, PrisonerInfoSet, RandomPrisonerPolicy, SwitchOnTwoSubsequent};
use sztorm_examples::prisoner::common::RewardTable;
use sztorm_examples::prisoner::domain::PrisonerAction::Betray;
use sztorm_examples::prisoner::domain::PrisonerDomain;
use sztorm_examples::prisoner::domain::PrisonerId::{Andrzej, Janusz};
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
    println!("Hello prisoners;");
    setup_logger(LevelFilter::Debug, &None).unwrap();

    let reward_table = RewardTable{
        cover_v_cover: 5,
        betray_v_cover: 10,
        betray_v_betray: 3,
        cover_v_betray: 1
    };


    let env_state = PrisonerEnvState::new(reward_table,  10);

    let (comm_env_0, comm_prisoner_0) = SyncCommEnv::new_pair();
    let (comm_env_1, comm_prisoner_1) = SyncCommEnv::new_pair();

    let mut prisoner0 = AgentGenT::new(
        Andrzej,
        PrisonerInfoSet::new(reward_table), comm_prisoner_0, CoverPolicy{});

    let mut prisoner1 = AgentGenT::new(
        Janusz,
        PrisonerInfoSet::new(reward_table), comm_prisoner_1, Forgive1Policy{});

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

    env.reset(PrisonerEnvState::new(reward_table,  10));
    let mut prisoner0 = prisoner0.transform_replace_policy(RandomPrisonerPolicy{});
    //let mut prisoner1 = prisoner1.do_change_policy(BetrayRatioPolicy{});
    let mut prisoner1 = prisoner1.transform_replace_policy(SwitchOnTwoSubsequent{});
    prisoner0.reset(PrisonerInfoSet::new(reward_table));
    prisoner1.reset(PrisonerInfoSet::new(reward_table));

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

    let prisoner0_betrayals = prisoner0.state().count_actions(Betray);
    let prisoner1_betrayals = prisoner1.state().count_actions(Betray);

    println!("Prisoner 0 betrayed {:?} times and Prisoner 1 betrayed {:?} times.", prisoner0_betrayals, prisoner1_betrayals);

    for elem in env.trajectory().list(){
        println!("{}", elem);
    }

    for trace in prisoner1.game_trajectory().list(){
        println!("{}", trace);
    }



    Ok(())
}