import argparse
import pickle
import sys
from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.agents import AgentDQN
from deep_dialog.usersims import RealUser
from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg
from deep_dialog.dialog_system import StateTracker

def main(user_message):
    # Arguments hardcodés pour le test
    parser = argparse.ArgumentParser()
    parser.add_argument('--diaact_nl_pairs', dest='diaact_nl_pairs', type=str,
                        default='./deep_dialog/data/dia_act_nl_pairs.v6.json',
                        help='path to the pre-defined dia_act&NL pairs')
    parser.add_argument('--nlg_model_path', dest='nlg_model_path', type=str,
                        default='./deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p',
                        help='path to model file')
    parser.add_argument('--nlu_model_path', dest='nlu_model_path', type=str,
                        default='./deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p',
                        help='path to the NLU model file')
    parser.add_argument('--trained_model_path', dest='trained_model_path', type=str,
                        default='./deep_dialog/checkpoints/agt_9_287_340_0.89200.p',
                        help='the path for trained model')

    args = parser.parse_args()
    params = vars(args)

    # Load data
    movie_kb_path = './deep_dialog/data/movie_kb.1k.p'
    with open(movie_kb_path, "rb") as f:
        movie_kb = pickle.load(f, encoding="latin1")
    act_set = text_to_dict('./deep_dialog/data/dia_acts.txt')
    slot_set = text_to_dict('./deep_dialog/data/slot_set.txt')
    dict_path = './deep_dialog/data/dicts.v3.p'
    movie_dictionary = pickle.load(open(dict_path, 'rb'))

    # Load models
    agent_params = {
        'max_turn': 20,
        'epsilon': 0,
        'agent_run_mode': 0,
        'agent_act_level': 0,
        'experience_replay_pool_size': 1000,
        'dqn_hidden_size': 60,
        'batch_size': 16,
        'gamma': 0.9,
        'predict_mode': False,
        'trained_model_path': params['trained_model_path'],
        'warm_start': 1,
        'cmd_input_mode': 0
    }
    
    agent = AgentDQN(movie_kb, act_set, slot_set, agent_params)
    user_sim = RealUser(movie_dictionary, act_set, slot_set)
    
    # Load NLU
    nlu_model = nlu()
    nlu_model.load_nlu_model(params['nlu_model_path'])
    
    # Load NLG
    nlg_model = nlg()
    nlg_model.load_nlg_model(params['nlg_model_path'])
    nlg_model.load_predefine_act_nl_pairs(params['diaact_nl_pairs'])

    agent.set_nlg_model(nlg_model)
    user_sim.set_nlg_model(nlg_model)
    agent.set_nlu_model(nlu_model)
    user_sim.set_nlu_model(nlu_model)
    dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, movie_kb)

    # Start dialog
    state_tracker = StateTracker(act_set, slot_set, movie_kb)
    state_tracker.initialize_episode()
    user_action = user_sim.initialize_episode()
    agent.initialize_episode()
    episode_over = False
    state_tracker.update(user_action=user_action)
    
    response_text = ""
    
    while not episode_over:
        state = state_tracker.get_state_for_agent()
        agent_action = agent.state_to_action(state)
        state_tracker.update(agent_action=agent_action)
        agent.add_nl_to_action(agent_action)
        sys_action = state_tracker.dialog_history_dictionaries()[-1]
        response_text = agent_action['act_slot_response']['nl']
        user_action, episode_over = user_sim.next(sys_action)
        reward = 0
        if not episode_over:
            state_tracker.update(user_action=user_action)
        agent.register_experience_replay_tuple(state, agent_action, reward,
                                               state_tracker.get_state_for_agent(), episode_over)
    
    print(response_text)
    return response_text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_message = sys.argv[1]
    else:
        user_message = ""
    
    main(user_message)
