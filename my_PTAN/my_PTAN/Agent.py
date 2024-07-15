import abc
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import typing as tt

from . import ActionSelectors

States = tt.Union[tt.List[np.ndarray], np.ndarray]
AgentStates = tt.List[tt.Any]
Preprocessor = tt.Callable[[States], torch.Tensor]

CPU_DEVICE = torch.device('cpu')
GPU_DEVICE = torch.device('cuda:0')


def default_states_preprocessor(states: States) -> torch.Tensor:
    """
    Convert list of states into the form suitable for model
    :param states: list of numpy arrays with states or numpy array
    :return: torch.Tensor
    """
    if isinstance(states, list):
        if len(states) == 1:
            np_states = np.expand_dims(states[0], 0)
        else:
            np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    else:
        np_states = states
    return torch.as_tensor(np_states)

def float32_preprocessor(states: States) -> torch.Tensor:
    np_states = np.array(states, dtype=np.float32)
    return torch.as_tensor(np_states)

class BaseAgent(abc.ABC):
    """
    Base agent class.
    """

    @abc.abstractmethod
    def __call__(self, states: States, 
                 agent_states: AgentStates) -> tt.Tuple[np.ndarray, AgentStates]:
        ...

    def initial_state(self) -> tt.Optional[tt.Any]:
        """
        idfk for what reason this function exists
        """
        return None

class NNAgent(BaseAgent):
    """
    NN based agent.
    """
    def __init__(self, 
                 model: nn.Module, 
                 action_selector: ActionSelectors.BaseActionSelector,
                 preprocessor: Preprocessor,
                 device: torch.device = GPU_DEVICE):
        """
        Constructor of base agent
        :param model: model to be used
        :param action_selector: action selector
        :param device: device for tensors
        :param pp: states preprocessor
        """
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.preprocessor = preprocessor
    
    @abc.abstractmethod
    def _net_filter(self, net_out: tt.Any, agent_states: AgentStates) -> \
            tt.Tuple[torch.Tensor, AgentStates]:
        """
        Internal method, processing network output and states into selector's input and new states
        :param net_out: output from the network
        :param agent_states: agent states
        :return: tuple with tensor to be fed into selector and new states
        """
        ...
    
    @torch.no_grad
    def __call__(self,
                 states: States,
                 agent_states: AgentStates = None) -> tt.Tuple[np.ndarray, AgentStates]:
        
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states): # why do we need this if???
                states = states.to(self.device)
        q_v = self.model(states)
        q_v, new_states = self._net_filter(q_v, agent_states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, new_states


class DQNAgent(NNAgent):
    def __init__(self, 
                 model: nn.Module, 
                 action_selector: ActionSelectors.BaseActionSelector,
                 preprocessor: Preprocessor,
                 device: torch.device = GPU_DEVICE):
        super(DQNAgent, self).__init__(model=model, 
                                       action_selector=action_selector, 
                                       preprocessor=preprocessor, 
                                       device=device)

        def _net_filter(self, net_out: tt.Any, agent_states: AgentStates) -> tt.Tuple[torch.Tensor, AgentStates]:
            assert torch.is_tensor(net_out)
            return net_out, agent_states

class TargetNet:
    def __init__(self, model: nn.Module):
        self.model = model
        self.target_model = copy.deepcopy(model)
    
    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def alpha_sync(self, alpha: float):
        """
        For smooth blending of parameters.
        """
        try:
            assert 0.0 <= alpha <= 1.0
        except AssertionError as e:
            print(f'Alpha not in [0,1] or not float. {e}')
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k]*alpha + v*(1-alpha)
        self.target_model.load_state_dict(tgt_state)

class PolicyAgent(NNAgent):
    def __init__(self, 
                 model: nn.Module, 
                 action_selector: ActionSelectors.BaseActionSelector,
                 preprocessor: Preprocessor,
                 apply_softmax: bool = False,
                 device: torch.device = GPU_DEVICE):
        super(DQNAgent, self).__init__(model=model, 
                                       action_selector=action_selector, 
                                       preprocessor=preprocessor, 
                                       device=device)
        self.apply_softmax = apply_softmax

    def _net_filter(self, net_out: tt.Any, agent_states: AgentStates) -> tt.Tuple[torch.Tensor, AgentStates]:
        try:
            assert torch.is_tensor(net_out)
        except AssertionError as e:
            print(f'net_out is not a torch.Tensor. {e}')
        if self.apply_softmax:
            return F.softmax(net_out, dim=1), agent_states
        return net_out, agent_states
            
class ActorCriticAgent(NNAgent):
    def __init__(self):
        raise NotImplemented
        return
