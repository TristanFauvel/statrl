
import sys
from six import StringIO
from gymnasium import utils
import string

class TextRenderer:
    """Print each MDP step to stdout as a colourized row of states.
 
    The current state is highlighted red and every state reachable from it in
    one step blue, alongside the action played and the reward it returned.

    Attributes
    ----------
    started : bool
        Whether the header has been printed yet; emitted lazily on the first
        :meth:`render`.
    """

    def __init__(self):
        self.started = False

    def start(self, env) -> None:
        """Print the header naming the environment, its actions, and the legend.

        Parameters
        ----------
        env : object
            The environment being rendered.
        """
        self.outfile = sys.stdout
        self.outfile.write("Environment: " + str(env.displayname) + "\n")
        self.outfile.write("Actions: "+ str(self._nameActions(env)) + "\n")
        self.outfile.write("Legend: Red=current state, Blue=possible next states\n")
        self.outfile.write("-"*30+"\n")

    def stop(self, env) -> None:
        """Print the closing rule at the end of a rendered run.

        Parameters
        ----------
        env : object
            The environment being rendered.
        """
        self.outfile.write("-"*30+"\n")

    def _nameActions(self, env) -> str:
        return string.ascii_uppercase[:env.nA]


    def render(self,env,last):
        """Print one step: the action, the reward, and the state row.

        Parameters
        ----------
        env : DiscreteMDP
            The environment being rendered.
        last : tuple of (int, int or None, float)
            The ``(state, action, reward)`` triple recorded by the
            environment. A ``None`` action means no step has been taken yet,
            and only the state row is printed.
        """
        current, lastaction, lastreward = last
        if (not self.started):
            self.start(env)
            self.started = True

        # Print the MDP in text mode.
        # Red  = current state
        # Blue = all states accessible from current state (by playing some action)

        desc = [str(s) for s in env.states]

        desc[current] = utils.colorize(desc[current], "red", highlight=True)
        for a in env.actions:
            for ssl in env.P[current][a]:
                if (ssl[0] > 0):
                    desc[ssl[1]] = utils.colorize(desc[ssl[1]], "blue", highlight=True)


        actionNames = self._nameActions(env)
        #print(f"\t{current},{lastaction},{lastreward}")
        if lastaction is not None:
            self.outfile.write(f"({actionNames[lastaction % 26]})\tr={lastreward:0.2f}\t")
            self.outfile.write("".join(desc))
            self.outfile.write("\n")
        else:
            self.outfile.write("\t\t\t")
            self.outfile.write("".join(desc))
            self.outfile.write("\n")