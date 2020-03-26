"""
Modified from https://github.com/vermouth1992/drl-portfolio-management/blob/master/src/environment/portfolio.py
"""
import gym
import gym.spaces
from pprint import pprint
import numpy as np
import csv

EPS = 1e-8


class PortfolioEnv(gym.Env):
    """ This class creates the financial market environment that the Agent interact with.

    It extends the OpenAI Gym environment https://gym.openai.com/.
    """

    def __init__(self, steps=730, trading_cost=0.0025, time_cost=0.00, window_length=7, start_idx=0,
                 sample_start_date=None):
        """An environment for financial portfolio management.

        Args:
            steps (int):  Steps in episode.
            trading_cost (float):  Cost of trade as a fraction.
            time_cost (float):  Cost of holding as a fraction.
            window_length (int):  How many past observations to return.
            start_idx (int):  The number of days from '2012-08-13' of the dataset.
            sample_start_date (None | str)  The sampling start date , e.g. '2012-08-13', random if `None`.
        """

        # Read the stock data
        root = os.path.split(os.path.dirname(__file__))[0]
        prices = pd.read_pickle(os.path.join(root, 'stock-data-clean.pkl'))
        open_close = np.dstack((prices.values[:-1].T, prices.values[1:].T))
        tickers = data.columns.tolist()
        n_tickers = len(tickers)
        self.dates = prices.index.tolist()

        # Read the signals
        signals = pd.read_pickle(os.path.join(root, 'signals.pkl'))
        n_signals = signals.shape[1]

        self.window_length = window_length
        self.num_stocks = n_tickers
        self.start_idx = start_idx
        self.csv_file = CSV_DIR

        self.src = DataGenerator(prices=open_close, tickers=tickers, max_steps=steps, window_length=window_length,
                                 start_idx=start_idx, start_date=sample_start_date, signals=signals)
        self.sim = PortfolioSim(tickers=tickers, trading_cost=trading_cost, time_cost=time_cost, max_steps=steps)

        # Define the action space as the portfolio weights [cash_bias,w1,w2...] where wn are [0, 1] for each asset
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(n_tickers + 1,), dtype=np.float32)

        # Define the observation space, which are the signals
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(n_signals, window_length), dtype=np.float32)

    # -----------------------------------------------------------------------------------
    def step(self, action):
        """Step the environment.

        Args:
            action (np.array):  The desired portfolio weights [w0...].

        Returns:
            state
            reward
            bool:  Indicates if the simulation is complete.
            dict:  Debugging information.
        """
        np.testing.assert_almost_equal(action.shape, (len(self.sim.asset_names) + 1,))

        # normalise just in case
        weights = np.clip(action, 0, 1)
        weights /= (weights.sum() + EPS)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)  # so if weights are all zeros we normalise to [1,0...]

        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(np.sum(weights), 1.0, 3, err_msg='weights must sum to 1. action="%s"' % weights)

        state, done1, ground_truth_obs = self.src.take_step()

        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation_concat = np.concatenate((cash_observation, observation), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        # relative price vector of last observation day (close/open)
        close_price_vector = observation_concat[:, -1, 1]
        open_price_vector = observation_concat[:, -1, 0]
        y1 = close_price_vector / open_price_vector
        reward, info, done2 = self.sim._step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        info['date'] = self.dates(self.start_idx + self.src.idx + self.src.step)
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs

        self.infos.append(info)

        if done1:
            # Save it to file
            keys = self.infos[0].keys()
            with open(self.csv_file, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.infos)
        return state, reward, done1 or done2, info

    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        observation, ground_truth_obs = self.src.reset()
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        info['next_obs'] = ground_truth_obs
        return observation


class DataGenerator(object):
    """Acts as data provider for each new episode."""

    def __init__(self, prices, signals, tickers, max_steps=730, window_length=50, start_idx=0, start_date=None):
        """
        Args:
            prices (np.array): [n_tickers, n_days, [open, close]] Stock price history.
            signals (np.array): [n_signals, n_days]  Signals to drive decisions.
            tickers (list):  Tickers of the stocks.
            max_steps (int):  The total number of steps to simulate, default is 2 years
            window_length (int):  Observation window, must be less than 50
            start_date (None | str):  The date to start, e.g. '2012-08-13'. Random if `None`.
        """
        assert prices.shape[0] == len(tickers), 'Number of stock is not consistent'

        self.tickers = tickers
        self.prices = prices.copy()
        self.idx = 0
        self.signals = signals.copy()
        self.start_idx = start_idx
        self.start_date = start_date
        self.step = 0
        self.max_steps = max_steps + 1
        self.window_length = window_length

        # Internal parameters
        self._prices = prices.copy()
        self._signals = signals.copy()

    def take_step(self):
        """Generate the current and next state of the environment."""
        self.step += 1
        state = self.signals[:, self.step:self.step + self.window_length]
        next_state = self.prices[:, self.step + self.window_length:self.step + self.window_length + 1]

        done = self.step >= self.max_steps
        return state, done, next_state

    def reset(self):
        """Resets the environment."""

        self.step = 0
        if self.start_date is None:
            self.idx = np.random.randint(low=self.window_length, high=self.prices.shape[1] - self.max_steps)
        else:
            # compute index corresponding to start_date for repeatable sequence
            self.idx = np.where(self.dates == self.start_date)[0][0] - self.start_idx
            error_text = 'Must have Windows length before start date and simulation steps after.'
            assert self.window_length <= self.idx <= self._prices.shape[1] - self.max_steps, error_text

        self.signals = self._signals[:, self.idx - self.window_length:self.idx + self.max_steps + 1, :4]

        state = self.signals[:, self.step:self.step + self.window_length, :].copy()
        next_state = self.signals[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()
        return state, next_state


class PortfolioSim(object):
    """ The environment simulator.  Determines the rewards and new state based on the actions and price history."""

    def __init__(self, tickers=(), max_steps=730, trading_cost=0.0025, time_cost=0.0):
        self.tickers = tickers
        self.cost = trading_cost
        self.time_cost = time_cost
        self.max_steps = max_steps
        self.infos = []
        self.w0 = np.array([1.0] + [0.0] * len(self.tickers))
        self.p0 = 1.0

    def _step(self, w1, y1):
        """Step in the environment. Numbered equations are from https://arxiv.org/abs/1706.10059

        Args:
            w1 (np.array):   New portfolio weights, e.g. [0.1,0.9,0.0].
            y1 (np.array):   Price relative vector also called return, e.g. [1.0, 0.9, 1.1].
        """
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        w0 = self.w0
        p0 = self.p0

        dw1 = (y1 * w0) / (np.dot(y1, w0) + EPS)  # (eq7) weights evolve into
        mu1 = self.cost * (np.abs(dw1 - w1)).sum()  # (eq16) cost to change portfolio
        assert mu1 < 1.0, 'Cost is larger than current holding'

        p1 = p0 * (1 - mu1) * np.dot(y1, w1)  # (eq11) final portfolio value
        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding
        p1 = np.clip(p1, 0, np.inf)  # no shorts

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + EPS) / (p0 + EPS))  # log rate of return
        reward = r1 / self.max_steps * 1000.  # (22) average logarithmic accumulated return
        # remember for next step
        self.w0 = w1
        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        done = bool(p1 == 0)

        info = {"reward": reward, "log_return": r1, "portfolio_value": p1, "return": y1.mean(), "rate_of_return": rho1,
                "weights_mean": w1.mean(), "weights_std": w1.std(), "cost": mu1}
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.w0 = np.array([1.0] + [0.0] * len(self.tickers))
        self.p0 = 1.0
