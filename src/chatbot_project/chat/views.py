from django.shortcuts import render, redirect
from .models import Conversation
from .forms import MessageForm
from deep_dialog.dialog_system import DialogManager, text_to_dict, StateTracker
from deep_dialog.agents import AgentDQN
from deep_dialog.usersims import RealUser
from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Charger les modèles et les données comme dans le script fourni
movie_kb_path = '../deep_dialog/data/movie_kb.1k.p'
with open(movie_kb_path, "rb") as f:
    movie_kb = pickle.load(f, encoding="latin1")

act_set = text_to_dict('../deep_dialog/data/dia_acts.txt')
slot_set = text_to_dict('../deep_dialog/data/slot_set.txt')
dict_path = '../deep_dialog/data/dicts.v3.p'
movie_dictionary = pickle.load(open(dict_path, 'rb'))

nlg_model_path = '../deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p'
diaact_nl_pairs = '../deep_dialog/data/dia_act_nl_pairs.v6.json'
nlu_model_path = '../deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p'

nlg_model = nlg()
nlg_model.load_nlg_model(nlg_model_path)
nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs)

nlu_model = nlu()
nlu_model.load_nlu_model(nlu_model_path)

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
    'trained_model_path': os.path.join(BASE_DIR, '..', 'deep_dialog', 'checkpoints', 'agt_9_287_340_0.89200.p'),
    'warm_start': 1,
    'cmd_input_mode': 0
}

agent = AgentDQN(movie_kb, act_set, slot_set, agent_params)
agent.set_nlg_model(nlg_model)
agent.set_nlu_model(nlu_model)

user_sim = RealUser(movie_dictionary, act_set, slot_set)
user_sim.set_nlg_model(nlg_model)
user_sim.set_nlu_model(nlu_model)

dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, movie_kb)

def chat_view(request):
    if request.method == "POST":
        form = MessageForm(request.POST)
        if form.is_valid():
            user_message = form.cleaned_data['message']
            state_tracker = StateTracker(act_set, slot_set, movie_kb)
            state_tracker.initialize_episode()
            user_action = user_sim.initialize_episode(user_message)
            agent.initialize_episode()
            state_tracker.update(user_action=user_action)
            state = state_tracker.get_state_for_agent()
            agent_action = agent.state_to_action(state)
            state_tracker.update(agent_action=agent_action)
            agent.add_nl_to_action(agent_action)
            sys_action = state_tracker.dialog_history_dictionaries()[-1]
            user_action, episode_over = user_sim.next(sys_action, user_message)
            bot_response = agent_action['act_slot_response']['nl']
            
            # Save conversation
            Conversation.objects.create(user_message=user_message, bot_response=bot_response)
            
            return redirect('chat:chat_view')
    else:
        form = MessageForm()
    
    conversations = Conversation.objects.all()
    return render(request, 'chat/chat_view.html', {'form': form, 'conversations': conversations})

def reset_conversation(request):
    if request.method == "POST":
        Conversation.objects.all().delete()  # Efface toutes les conversations
        return redirect('chat:chat_view')



