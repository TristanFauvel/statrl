



import numpy as np
from statrl.settings.markovdecisionprocess.discrete_nostructure.environment import DiscreteMDP
from statrl.settings.markovdecisionprocess.discrete_nostructure.agent import  MDPAgent


from statrl.settings.markovdecisionprocess.discrete_nostructure.renderers.textRenderer import TextRenderer
from statrl.settings.markovdecisionprocess.discrete_nostructure.renderers.htmlRenderer import HTMLRenderer

from statrl.experiments.onerun import Interaction

class MDPInteraction(Interaction):
    """Interaction loop for the discrete MDP setting.

    Drives ``play(state)`` -> ``step(action)`` -> ``update(state, action,
    reward, next_state)`` for a fixed number of rounds.

    On a terminal transition the environment is reset and the run continues,
    so ``horizon`` counts *steps*, not episodes.
    """

    def run(self, env: DiscreteMDP, learner: MDPAgent, horizon: int) -> np.ndarray:
        """Run one interaction and return its cumulative expected score.

        Parameters
        ----------
        env : DiscreteMDP
            The MDP instance. Reset at the start, and again after any terminal
            transition.
        learner : MDPAgent
            The agent, reset with the initial state.
        horizon : int
            Number of steps to play.

        Returns
        -------
        ndarray of shape (horizon,)
            Cumulative sum of the *expected* rewards of the visited
            state-action pairs.
        """
        observation, info = env.reset()
        learner.reset(observation)

        steps_scores = np.empty(horizon)
        for t in range(horizon):
            state = observation
            action = learner.play(state)  # Get action
            observation, reward, done, truncated, info = env.step(action)
            learner.update(state, action, reward, observation)  # Update learners

            steps_scores[t] = env.expected_reward(state,action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                observation, info = env.reset()


        return np.cumsum(steps_scores)

    def renderrun(self, env: DiscreteMDP, learner: MDPAgent, horizon: int) -> None:
        """Run one interaction, printing each step to stdout.

        Parameters
        ----------
        env : DiscreteMDP
            The MDP instance. Its ``renderers`` list is overwritten with a
            text renderer.
        learner : MDPAgent
            The agent.
        horizon : int
            Number of steps to play.
        """
        env.renderers= [TextRenderer(),HTMLRenderer()]
        observation, info = env.reset()
        learner.reset(observation)

        env.render()
        for t in range(horizon):
            state = observation
            action = learner.play(state)  # Get action
            observation, reward, done, truncated, info = env.step(action)
            learner.update(state, action, reward, observation)  # Update learners

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                observation, info = env.reset()

            env.render()

        env.close()


    @property
    def plotlabels(self):
        """tuple of (str, str): Axis labels ``(x, y)`` for the regret plots."""
        return ("Time step", "Regret")


