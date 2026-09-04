
import sys
from gymnasium import utils
import string

class GridworldWithWallRenderer:
    """Print a walled gridworld to stdout as an ASCII map.

    Draws the grid each step, marking the current cell red, walls ``X``, and
    goal cells ``G``.

    Attributes
    ----------
    started : bool
        Whether the header has been printed yet.

    See Also
    --------
    GridworldRenderer : The counterpart for grids without walls.
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
        self.outfile.write("Actions: " + str(self._nameActions(env)) + "\n")
        self.outfile.write("Legend: Red=current state, X=wall, G=goal state\n")
        self.outfile.write("-" * 30 + "\n")

    def stop(self, env) -> None:
        """Print the closing rule at the end of a rendered run.

        Parameters
        ----------
        env : object
            The environment being rendered.
        """
        self.outfile.write("-" * 30 + "\n")


    def _nameActions(self, env) -> str:
        return string.ascii_uppercase[:env.nA]

    def render(self,env,last):
        """Print the grid with the current cell highlighted.

        Parameters
        ----------
        env : object
            The gridworld being rendered.
        current : int
            Index of the current cell.
        lastaction : int or None
            Action just played, or ``None`` before the first step.
        lastreward : float
            Reward just observed.
        """

        current, lastaction, lastreward = last
        if (not self.started):
            self.start(env)
            self.started = True

        # Print the MDp in text mode.
        # Red  = current state
        # Blue = all states accessible from current state (by playing some action)
        #outfile = sys.stdout
        #outfile = StringIO() if mode == 'ansi' else sys.stdout

        symbols = {0.: 'X', 1.: '.', 2.: 'G'}
        desc = [[symbols[c] for c in line] for line in env.maze]
        row, col = env.from_s(current)
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        desc.append(" \t\tr=" + str(lastreward))
        if lastaction is not None:
            self.outfile.write("  ({})\n".format(env.nameActions[lastaction]))
        else:
            self.outfile.write("\n")
        self.outfile.write("\n".join(''.join(line) for line in desc) + "\n")





class GridworldRenderer:
    """Print a gridworld without walls to stdout as an ASCII map.

    Draws the grid each step, marking the current cell red and goal cells
    ``G``.

    Attributes
    ----------
    started : bool
        Whether the header has been printed yet.

    See Also
    --------
    GridworldWithWallRenderer : The counterpart for grids with walls.
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
        self.outfile.write("Actions: " + str(self._nameActions(env)) + "\n")
        self.outfile.write("Legend: Red=current state, X=wall, G=goal state\n")
        self.outfile.write("-" * 30 + "\n")

    def stop(self, env) -> None:
        """Print the closing rule at the end of a rendered run.

        Parameters
        ----------
        env : object
            The environment being rendered.
        """
        self.outfile.write("-" * 30 + "\n")


    def _nameActions(self, env) -> str:
        return string.ascii_uppercase[:env.nA]

    def render(self,env,last):
        """Print the grid with the current cell highlighted.

        Parameters
        ----------
        env : object
            The gridworld being rendered.
        current : int
            Index of the current cell.
        lastaction : int or None
            Action just played, or ``None`` before the first step.
        lastreward : float
            Reward just observed.
        """

        current, lastaction, lastreward = last
        if (not self.started):
            self.start(env)
            self.started = True

        # Print the MDP in text mode.
        # Red  = current state
        # Blue = all states accessible from current state (by playing some action)
        #outfile = sys.stdout
        #outfile = StringIO() if mode == 'ansi' else sys.stdout

        symbols = {0.: 'X', 1.: '.', 2.: 'G'}
        desc = [[symbols[c] for c in line] for line in env.maze]
        row, col = env.from_s(env.mapping[current])
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)


        #desc.append(" \t\tr=" + str(lastreward))
        if lastaction is not None:
            self.outfile.write("\t({})\tr={}\n\n".format(env.nameActions[lastaction],str(lastreward)))
        else:
            self.outfile.write(" \n")

        self.outfile.write("\n".join(''.join(line) for line in desc) + "\n")


