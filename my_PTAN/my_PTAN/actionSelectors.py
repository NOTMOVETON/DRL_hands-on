import abc
import numpy as np
import typing as tt

class BaseActionSelector(abc.ABC):
    '''
    :)
    '''
    @abc.abstractmethod
    def __call__(self, scores: np.ndarray) -> np.ndarray:
        pass


class ArgmaxActionSelector(BaseActionSelector):
    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return np.argmax(scores, axis=1)

class EpsilonGreedyActionSelector(BaseActionSelector):
    def __init__(self, epsilon: float = 0.05):
        self._epsilon = epsilon
    
    @property
    def epsilon(self) -> float:
        return self._epsilon
    
    @epsilon.setter
    def epsilon(self, new_epsilon):
        if (new_epsilon > 1.0) or (new_epsilon < 0.0):
            raise ValueError('Wrong new_epsilon value. Must be in [0,1].')
        self._epsilon = new_epsilon
    
    def __call__(self, scores: np.ndarray) -> np.ndarray:
        batch_size, actions_n = scores.shape
        actions = np.argmax(scores, axis=1)
        mask = np.random.random(batch_size)
        mask = mask[mask<self._epsilon]
        # get random action(discrete env) in size of total mask<eps 
        random_actions = np.random.choice(actions_n, sum(mask))
        actions[mask] = random_actions
        return actions



class EpsilonTracker(BaseActionSelector):
    def __init__(self, selector: EpsilonGreedyActionSelector,
                 eps_start: tt.Union[int, float],
                 eps_final: tt.Union[int, float],
                 eps_frames: int):
        self.selector = selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames
    
    def update(self, step: int):
        eps = self.eps_start - step/self.eps_frames
        eps = max(eps, self.eps_final)
        self.selector.epsilon = eps

class PropabilityActionSelector(BaseActionSelector):
    def __call__(self, probs: np.ndarray) -> np.ndarray:
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)
