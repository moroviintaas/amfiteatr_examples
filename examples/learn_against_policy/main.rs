mod options;
mod plots;

use std::{thread};
use std::marker::PhantomData;
use std::path::{Path};
use std::sync::{Arc, Mutex};
use log::{debug, info};
use tch::{Device, nn, Tensor};
use tch::nn::{Adam, VarStore};
use amfi_rl::tensor_repr::{WayToTensor};
use amfi_rl::torch_net::{A2CNet, NeuralNetTemplate, TensorA2C};
use clap::{Parser};
use plotters::style::colors;
use amfi::agent::*;
use amfi::comm::{EnvMpscPort, SyncCommAgent};
use amfi::env::{AutoEnvironmentWithScores, ReseedEnvironment, ScoreEnvironment, TracingEnv};
use amfi::env::generic::TracingEnvironment;
use amfi::error::AmfiError;
use amfi_classic::agent::{OwnHistoryInfoSet, OwnHistoryInfoSetNumbered, OwnHistoryTensorRepr, SwitchAfterTwo};
use amfi_classic::domain::{AgentNum, ClassicGameDomain, ClassicGameDomainNumbered,  UsizeAgentId};
use amfi_classic::env::PairingState;
use amfi_classic::policy::ClassicMixedStrategy;
use amfi_classic::SymmetricRewardTableInt;
use amfi_rl::actor_critic::ActorCriticPolicy;
use amfi_rl::{LearningNetworkPolicy, TrainConfig};
use crate::options::EducatorOptions;
use crate::options::SecondPolicy;
use crate::plots::{plot_many_payoffs, Series};


pub struct ModelElements<ID: UsizeAgentId, Seed>{
    pub environment: Arc<Mutex<dyn AutoEnvironmentWithScores<ClassicGameDomain<ID>>>>,
    agents: [Arc<Mutex<dyn AutomaticAgentRewarded<ClassicGameDomain<ID>>>>;2],
    seed: PhantomData<Seed>,
}

pub fn setup_logger(options: &EducatorOptions) -> Result<(), fern::InitError> {
    let dispatch  = fern::Dispatch::new()

        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level(options.log_level)
        .level_for("amfi_examples", options.log_level)
        .level_for("amfi", options.log_level_amfi);

        match &options.log_file{
            None => dispatch.chain(std::io::stdout()),
            Some(f) => dispatch.chain(fern::log_file(f)?)
        }

        .apply()?;
    Ok(())
}
type Domain = ClassicGameDomain<AgentNum>;
type A2C = ActorCriticPolicy<D, OwnHistoryInfoSetNumbered, OwnHistoryTensorRepr>;


pub fn run_game(
    env: &mut (impl AutoEnvironmentWithScores<Domain> + Send + ReseedEnvironment<Domain, ()>),
    agent0: &mut (impl MultiEpisodeAgent<Domain, ()> + AutomaticAgentRewarded<Domain> + Send + ReseedAgent<Domain, ()>),
    //agent1: &mut (impl MultiEpisodeAgent<Domain, ()> + AutomaticAgentRewarded<Domain> + Send + ReseedAgent<Domain, ()>)
    agent1: &mut Box<dyn ModelAgent<Domain, (), OwnHistoryInfoSetNumbered>>
    )
    -> Result<(), AmfiError<Domain>>{

    thread::scope(|s|{
        s.spawn(||{
            env.reseed(());
            env.run_with_scores().unwrap();
        });
        s.spawn(||{
            agent0.reseed(());
            agent0.run_episode_rewarded(()).unwrap()
        });
        s.spawn(||{
            //let mut g = agent1.lock().unwrap();
            //g.run_episode_rewarded(()).unwrap()

            agent1.reseed(());
            agent1.run_episode_rewarded(()).unwrap()
        });
    });
    Ok(())

}
/*
pub trait CustomLearningAgent<DP: DomainParameters>:
    MultiEpisodeAgent<DP> + NetworkLearningAgent<DP> + AutomaticAgentBothPayoffs<DP>{}
pub trait CustomNotLearningAgent<DP: DomainParameters>:
    MultiEpisodeAgent<DP> + AutomaticAgentBothPayoffs<DP>{}

impl<DP: DomainParameters, T: MultiEpisodeAgent<DP> + NetworkLearningAgent<DP> + AutomaticAgentBothPayoffs<DP>>
CustomLearningAgent<DP> for T{}

impl<DP: DomainParameters, T: MultiEpisodeAgent<DP>  + AutomaticAgentBothPayoffs<DP>>
CustomNotLearningAgent<DP> for T{}
*/
pub enum AgentWrap{
    //Learning(Arc<Mutex<dyn NetworkLearningAgent<InfoSetType=(), Policy=()>>>),
    //Simple(Arc<Mutex<dyn Au>>)
}
type D = ClassicGameDomainNumbered;
type C = SyncCommAgent<D>;
type IS = OwnHistoryInfoSet<AgentNum>;


fn main() -> Result<(), AmfiError<ClassicGameDomain<AgentNum>>>{

    let args = EducatorOptions::parse();
    setup_logger(&args).unwrap();
    let device = Device::Cpu;
    type Domain = ClassicGameDomainNumbered;
    let number_of_players = 2;



    let tensor_repr = OwnHistoryTensorRepr::new(args.number_of_rounds);

    let input_size = tensor_repr.desired_shape().iter().product();

    let mut payoffs_0 = Vec::with_capacity(args.epochs + 1);
    let mut payoffs_1 = Vec::with_capacity(args.epochs + 1);
    let mut custom_payoffs_1 = Vec::with_capacity(args.epochs + 1);
    //let mut opti_payoffs_1 = Vec::with_capacity(args.epochs + 1);



    let mut env_adapter = EnvMpscPort::new();
    let comm0 = env_adapter.register_agent(0).unwrap();
    let comm1 = env_adapter.register_agent(1).unwrap();

    let reward_table = SymmetricRewardTableInt::new(5, 1, 10, 3);


    let net_template = NeuralNetTemplate::new(|path|{
        let seq = nn::seq()
            .add(nn::linear(path / "input", input_size, 512, Default::default()))
            //.add(nn::linear(path / "h1", 256, 256, Default::default()))
            .add(nn::linear(path / "hidden1", 512, 512, Default::default()))
            .add(nn::linear(path / "hidden2", 512, 512, Default::default()))
            .add_fn(|xs|xs.relu());
            //.add(nn::linear(path / "h2", 512, 512, Default::default()));
        let actor = nn::linear(path / "al", 512, 2, Default::default());
        let critic =  nn::linear(path / "ac", 512, 1, Default::default());
        {move |input: &Tensor|{
            let xs = input.to_device(device).apply(&seq);
            TensorA2C{critic: xs.apply(&critic), actor: xs.apply(&actor)}
        }}
    });







    let env_state_template = PairingState::new_even(number_of_players, args.number_of_rounds, reward_table.into()).unwrap();
    let mut environment = TracingEnvironment::new(env_state_template.clone(), env_adapter);


    let net0 = A2CNet::new(VarStore::new(device), net_template.get_net_closure());
    let opt0 = net0.build_optimizer(Adam::default(), 1e-4).unwrap();
    let normal_policy = ActorCriticPolicy::new(net0, opt0, tensor_repr, TrainConfig {gamma: 0.99});
    let state0 = OwnHistoryInfoSet::new(0, reward_table.into());
    let mut agent_0 = AgentGenT::new(state0, comm0, normal_policy);


    let state1 = OwnHistoryInfoSet::new(1, reward_table.into());
    //let test_policy = ClassicPureStrategy::new(ClassicAction::Defect);

    let net1 = A2CNet::new(VarStore::new(device), net_template.get_net_closure());
    let opt1 = net1.build_optimizer(Adam::default(), 1e-4).unwrap();
    //let policy1 = ActorCriticPolicy::new(net1, opt1, tensor_repr, TrainConfig {gamma: 0.99});
    //let mut agent_1 = AgentGenT::new(state1, comm1, Arc::new(Mutex::new(policy1)));

    //AgentGenT::new(state1, comm1, policy1);
    let mut agent_1: Box<dyn ModelAgent<D, (), OwnHistoryInfoSetNumbered, >> = match args.policy{
        SecondPolicy::Mixed => {
            Box::new(AgentGenT::new(state1, comm1, ClassicMixedStrategy::new(args.defect_proba as f64)))
        }
        SecondPolicy::SwitchTwo => {Box::new(AgentGenT::new(state1, comm1, SwitchAfterTwo{}))}
    };


    //evaluate on start
    let mut scores = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..100{
        debug!("Plaing round: {i:} of initial simulation");
        //let mut agent_1_guard = agent_1.lock().unwrap();
        run_game(&mut environment, &mut agent_0, &mut agent_1)?;
        scores[0].push(agent_0.current_universal_score()) ;
        scores[1].push(agent_1.current_universal_score());
        //scores[2].push(reward_f(agent_1.current_subjective_score()) as i64);


    }
    let avg = [scores[0].iter().sum::<i64>()/(scores[0].len() as i64),
            scores[1].iter().sum::<i64>()/(scores[1].len() as i64),
            scores[2].iter().sum::<i64>()/(scores[2].len() as i64)];
        info!("Average scores: 0: {}\t1:{}", avg[0], avg[1]);

    payoffs_0.push(avg[0] as f32);
    payoffs_1.push(avg[1] as f32);
    custom_payoffs_1.push(avg[2] as f32);


    for e in 0..args.epochs{
        agent_0.clear_episodes();
        agent_1.clear_episodes();
        info!("Starting epoch {e:}");
        for _g in 0..args.batch_size{
            run_game(&mut environment, &mut agent_0, &mut agent_1)?;
        }
        let trajectories_0 = agent_0.take_episodes();
        //let trajectories_1 = agent_1.take_episodes();
        agent_0.policy_mut().train_on_trajectories_env_reward(&trajectories_0[..])?;



        let mut scores = [Vec::new(), Vec::new(), Vec::new()];
        for i in 0..100{
            debug!("Plaing round: {i:} of initial simulation");
            run_game(&mut environment, &mut agent_0, &mut agent_1)?;
            scores[0].push(agent_0.current_universal_score());
            scores[1].push(agent_1.current_universal_score());
            //scores[2].push(reward_f(agent_1.current_subjective_score()) as i64);

        }

        let avg = [scores[0].iter().sum::<i64>() as f64 /(scores[0].len() as f64),
            scores[1].iter().sum::<i64>() as f64/(scores[1].len() as f64),
            scores[2].iter().sum::<i64>() as f64/(scores[2].len() as f64),
        ];
        debug!("Score sums: {scores:?}, of size: ({}, {}).", scores[0].len(), scores[1].len());
        info!("Average scores: 0: {}\t1: {}", avg[0], avg[1]);
        payoffs_0.push(avg[0] as f32);
        payoffs_1.push(avg[1] as f32);
        custom_payoffs_1.push(avg[2] as f32);
    }

    run_game(&mut environment, &mut agent_0, &mut agent_1)?;
    println!("{:?}", agent_0.take_episodes().last().unwrap().list().last().unwrap());



    println!("{}", environment.trajectory().list().last().unwrap());

    println!("Scores: 0: {},\t1: {}", environment.actual_score_of_player(&0), environment.actual_score_of_player(&1));


    //plot_payoffs(Path::new(format!("agent_0-{:?}-{:?}.svg", args.policy, args.number_of_rounds).as_str()), &payoffs_0[..]).unwrap();
    //plot_payoffs(Path::new(format!("agent_1-{:?}-{:?}.svg", args.policy, args.number_of_rounds).as_str()), &payoffs_1[..]).unwrap();

    let agent0_data = Series{
        data: payoffs_0,
        description: "Agent 0".to_string(),
        color: colors::RED,
    };
    let agent1_data = Series{
        data: payoffs_1,
        description: "Agent 1".to_string(),
        color: colors::BLUE,
    };

    let agent1_custom_data = Series{
        data: custom_payoffs_1,
        description: "Agent 1 - custom reward".to_string(),
        color: colors::GREEN,
    };



    let s_policy = match args.policy{
        SecondPolicy::Mixed => {format!("mixed-{:.02}", args.defect_proba)}
        SecondPolicy::SwitchTwo => {format!("switch2")}
    };


    plot_many_payoffs(Path::new(
        format!("results/payoffs-{}-{:?}-{}.svg",
                &s_policy.as_str(),
                args.number_of_rounds,
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"))
            .as_str(), ), &[agent0_data, agent1_data, agent1_custom_data]).unwrap();
    //plot_payoffs(Path::new(format!("custom-payoffs-{:?}-{:?}.svg", args.policy, args.number_of_rounds).as_str()), &agent1_custom_data ).unwrap();

    Ok(())
    //let standard_strategy =
}

