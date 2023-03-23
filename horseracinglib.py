from sys import prefix
import pandas as pd
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

# ----------------------------------------------------------------

class ProbabilityModel:
    def __init__(self, model_prefix='undef'):
        self.model_coefficients = None
        self.model_prefix = model_prefix
        self.model_probabilities = None
    
    def calculate_model_probabilities_for_single_race(self, runners_single_race):
        pass

    def calculate_model_probabilities_for_multiple_races(self, runners_multiple_races):
        self.model_probabilities = pd.merge(runners_multiple_races[['race_id', 'runner_id']], runners_multiple_races.groupby('race_id', group_keys=False).apply(self.calculate_model_probabilities_for_single_race), how='left', on=['race_id', 'runner_id'], validate='1:1')

class MultinomialLogitModel(ProbabilityModel):
    def __init__(self, coefficient_filename, model_prefix='undef'):
        super().__init__(model_prefix)
        self.model_coefficients = pd.read_csv(coefficient_filename)

    def calculate_model_probabilities_for_single_race(self, runners_single_race):
        utility = np.matmul(runners_single_race[self.model_coefficients.feature].to_numpy(), self.model_coefficients.coefficient.to_numpy())
        mod_prob = np.exp(utility)
        mod_prob = mod_prob / np.sum(mod_prob)
        return pd.DataFrame({'race_id': runners_single_race.race_id, 'runner_id': runners_single_race.runner_id, 'stall_number': runners_single_race.stall_number, 'win': runners_single_race.win, 'mod_prob': mod_prob})

class RandomChoiceModel(ProbabilityModel):
    def calculate_model_probabilities_for_single_race(self, runners_single_race):
        rng = default_rng()
        mod_prob = rng.uniform(low=0.0, high=1.0, size=len(runners_single_race))
        mod_prob = mod_prob / mod_prob.sum()
        return pd.DataFrame({'race_id': runners_single_race.race_id, 'runner_id': runners_single_race.runner_id, 'stall_number': runners_single_race.stall_number, 'win': runners_single_race.win, 'mod_prob': mod_prob})

class FavouriteChoiceModel(ProbabilityModel):
    def calculate_model_probabilities_for_single_race(self, runners_single_race):
        return pd.DataFrame({'race_id': runners_single_race.race_id, 'runner_id': runners_single_race.runner_id, 'stall_number': runners_single_race.stall_number, 'win': runners_single_race.win, 'mod_prob': runners_single_race.adj_mkt_prob})

# ----------------------------------------------------------------

class ProbabilityModelAssessment:
    BINS = ((0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 1.00))
    MARKERS = ['ro', 'g+', 'bx', 'y.']
    LINED_MARKERS = ['r-o', 'g-+', 'b-x', 'y-.']

    def __init__(self, model_list, abridged_runners_multiple_races):
        n = len(model_list)
        if n < 1 or n > 4:
            raise ValueError('Too few or too many models')
        elif n > len(np.unique([model.model_prefix for model in model_list])):
            raise ValueError('Common model prefixes')
        else:
            self.model_list = model_list
        
        if 'race_id' not in abridged_runners_multiple_races.columns or 'runner_id' not in abridged_runners_multiple_races.columns or 'win' not in abridged_runners_multiple_races.columns or 'adj_mkt_prob' not in abridged_runners_multiple_races.columns:
            raise ValueError('abridged_runners_multiple_races must contain race_id, runner_id, win, and adj_mkt_prob')
        else:
            self.assessment = abridged_runners_multiple_races

        for model in self.model_list:
            self.assessment = pd.merge(self.assessment, model.model_probabilities[['race_id', 'runner_id', 'mod_prob']].rename({'mod_prob': model.model_prefix + '_mod_prob'}, axis=1, inplace=False), how='left', on=['runner_id', 'race_id'], validate='1:1')
        self.assessment_has_been_performed = False

    def perform_assessment(self):
        self.assessment_has_been_performed = True
        
        self.win_perc = {}
        self.avg_prob = {}
        self.P1 = {}
        self.P2 ={}
        
        n = len(ProbabilityModelAssessment.BINS)
        for model in self.model_list:
            mod_prob_col = model.model_prefix + '_mod_prob'
                
            self.win_perc[model.model_prefix] = np.full(n, np.nan, dtype=np.float64)
            self.avg_prob[model.model_prefix] = np.full(n, np.nan, dtype=np.float64)
            self.P1[model.model_prefix] = np.full(n, np.nan, dtype=np.float64)
            self.P2[model.model_prefix] = np.full(n, np.nan, dtype=np.float64)
            
            i = 0
            for lo, hi in ProbabilityModelAssessment.BINS:
                cond1 = (self.assessment[mod_prob_col] >= lo)
                cond2 = cond1 & (self.assessment[mod_prob_col] < hi)
                cond3 = (self.assessment.win == 1) & cond1
                
                self.win_perc[model.model_prefix][i] = self.assessment.loc[cond2, 'win'].sum() / len(self.assessment[cond2])
                self.avg_prob[model.model_prefix][i] = self.assessment.loc[cond2, mod_prob_col].mean()
                
                self.P1[model.model_prefix][i] = self.assessment.loc[cond3, mod_prob_col].mean() # conditional mean model probability for winners
                self.P2[model.model_prefix][i] = ((self.assessment.loc[cond1, 'win'] - self.assessment.loc[cond1, mod_prob_col]) **2).sum() # conditional brier score for model probability
                
                i += 1

    def plot_model_probability_vs_win_percentage(self):
        if not self.assessment_has_been_performed:
            self.perform_assessment()
            
        plt.rcParams["figure.figsize"] = (14,12)

        fig, ax = plt.subplots()

        i = 0
        for model in self.model_list:
            ax.plot(self.avg_prob[model.model_prefix], self.win_perc[model.model_prefix], ProbabilityModelAssessment.MARKERS[i], label=model.model_prefix)
            i += 1
        
        ax.axline([0, 0], [1, 1])

        plt.xlabel("Model probability", fontsize=15)
        plt.ylabel("Win percentage", fontsize=15)
        plt.grid()
        plt.legend()

        plt.show()
    
    def plot_P1(self):
        if not self.assessment_has_been_performed:
            self.perform_assessment()
            
        plt.rcParams["figure.figsize"] = (14,12)

        fig, ax = plt.subplots()

        i = 0
        for model in self.model_list:
            ax.plot([lo for lo, hi in ProbabilityModelAssessment.BINS], self.P1[model.model_prefix], ProbabilityModelAssessment.LINED_MARKERS[i], label=model.model_prefix)
            i += 1

        plt.xlabel("Model probability", fontsize=15)
        plt.ylabel("Conditional mean probability for winners (P1)", fontsize=15)
        plt.grid()
        plt.legend()

        plt.show()
    
    def plot_P2(self):
        if not self.assessment_has_been_performed:
            self.perform_assessment()
            
        plt.rcParams["figure.figsize"] = (14,12)

        fig, ax = plt.subplots()

        i = 0
        for model in self.model_list:
            ax.plot([lo for lo, hi in ProbabilityModelAssessment.BINS], self.P2[model.model_prefix], ProbabilityModelAssessment.LINED_MARKERS[i], label=model.model_prefix)
            i += 1

        plt.xlabel("Model probability", fontsize=15)
        plt.ylabel("Conditional Brier score for winners (P2)", fontsize=15)
        plt.grid()
        plt.legend()

        plt.show()    

    def plot_model_probability_vs_market_probability(self):
        if not self.assessment_has_been_performed:
            self.perform_assessment()
            
        plt.rcParams["figure.figsize"] = (14,12)

        fig, axes = plt.subplots(2,2)

        i = 0
        for model in self.model_list:
            r, c = divmod(i, 2)
            axes[r][c].plot(self.assessment['adj_mkt_prob'], self.assessment[model.model_prefix + '_mod_prob'], 'b.', label=model.model_prefix)
            axes[r][c].axline([0, 0], [1, 1])

            axes[r][c].set_xlabel("Adjusted market probability", fontsize=15)
            axes[r][c].set_ylabel("Model probability", fontsize=15)
            
            axes[r][c].legend()
        
            i += 1
            
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)

    def show_diagnostic_plots(self):
        self.plot_model_probability_vs_win_percentage()
        self.plot_P1()
        self.plot_P2()
        self.plot_model_probability_vs_market_probability()

# ----------------------------------------------------------------

class WageringStrategy:
    def __init__(self, probability_model, strategy_algorithm, strategy_prefix='undef'):
        self.probability_model = probability_model
        self.strategy_algorithm = strategy_algorithm
        self.strategy_prefix = strategy_prefix + '(' + self.probability_model.model_prefix + ')' # incorporate model prefix into strategy prefix
        self.strategy_stakes_and_payoffs = None
    
    def calculate_strategy_stakes_and_payoffs_for_single_race(self, runners_single_race, recalculate_model_probabilities=True):
        if recalculate_model_probabilities:
            augmented_runners_single_race = pd.merge(runners_single_race, self.probability_model.calculate_model_probabilities_for_single_race(runners_single_race)[['race_id', 'runner_id', 'mod_prob']], how='left', on=['race_id', 'runner_id'], validate='1:1')
        else:
            augmented_runners_single_race = runners_single_race # called by calculate_strategy_stakes_and_payoffs_for_multiple_races()
        augmented_runners_single_race['strat_stake'] = self.strategy_algorithm(augmented_runners_single_race)
        augmented_runners_single_race['strat_payoff'] = augmented_runners_single_race.win * augmented_runners_single_race.strat_stake * augmented_runners_single_race.sp
        return pd.DataFrame({'race_id': augmented_runners_single_race.race_id, 'runner_id': augmented_runners_single_race.runner_id, 'stall_number': augmented_runners_single_race.stall_number, 'sp': augmented_runners_single_race.sp, 'win': augmented_runners_single_race.win, 'mod_prob': augmented_runners_single_race.mod_prob, 'strat_stake': augmented_runners_single_race.strat_stake, 'strat_payoff': augmented_runners_single_race.strat_payoff})

    def calculate_strategy_stakes_and_payoffs_for_multiple_races(self, runners_multiple_races):
        self.probability_model.calculate_model_probabilities_for_multiple_races(runners_multiple_races)
        augmented_runners_multiple_races = pd.merge(runners_multiple_races, self.probability_model.model_probabilities[['race_id', 'runner_id', 'mod_prob']], how='left', on=['race_id', 'runner_id'], validate='1:1')
        self.strategy_stakes_and_payoffs = augmented_runners_multiple_races.groupby('race_id', as_index=False, group_keys=False).apply(lambda rmr: self.calculate_strategy_stakes_and_payoffs_for_single_race(rmr, recalculate_model_probabilities=False))

# ----------------------------------------------------------------

class WageringStrategyAssessment:
    LINED_MARKERS = ['r-o', 'g-+', 'b-x', 'y-.']
    
    def __init__(self, strategy_list, abridged_runners_multiple_races):
        n = len(strategy_list)
        if n < 1 or n > 4:
            raise ValueError('Too few or too many strategies')
        elif n > len(np.unique([strategy.strategy_prefix for strategy in strategy_list])):
            raise ValueError('Common strategy prefixes')
        else:
            self.strategy_list = strategy_list
        
        if 'race_id' not in abridged_runners_multiple_races.columns or 'meeting_date' not in abridged_runners_multiple_races.columns or 'runner_id' not in abridged_runners_multiple_races.columns:
            raise ValueError('abridged_runners_multiple_races must contain race_id, meeting_date, and runner_id')
        else:
            self.assessment = abridged_runners_multiple_races
        
        for strategy in self.strategy_list:
            self.assessment = pd.merge(self.assessment, strategy.strategy_stakes_and_payoffs[['race_id', 'runner_id', 'mod_prob', 'strat_stake', 'strat_payoff']].rename({'mod_prob': strategy.strategy_prefix + '_mod_prob', 'strat_stake': strategy.strategy_prefix + '_strat_stake', 'strat_payoff': strategy.strategy_prefix + '_strat_payoff',}, axis=1, inplace=False), how='left', on=['runner_id', 'race_id'], validate='1:1')
        self.assessment_has_been_performed = False
    
    def perform_assessment(self):
        self.assessment_has_been_performed = True
        resorted_assessment = self.assessment.sort_values(['meeting_date'], inplace=False) # don't mess with self.assessment
        resorted_assessment['as_of'] = resorted_assessment['meeting_date'] + pd.offsets.MonthBegin(1)
            
        self.monthly_assessment = {}

        for strategy in self.strategy_list:
            self.monthly_assessment[strategy.strategy_prefix] = resorted_assessment.groupby(['as_of']).agg({strategy.strategy_prefix + '_strat_stake': 'sum', strategy.strategy_prefix + '_strat_payoff': 'sum'})
            self.monthly_assessment[strategy.strategy_prefix].reset_index(inplace=True)
            self.monthly_assessment[strategy.strategy_prefix].rename({strategy.strategy_prefix + '_strat_stake': 'strat_stake', strategy.strategy_prefix + '_strat_payoff': 'strat_payoff'}, axis=1, inplace=True)
            
            self.monthly_assessment[strategy.strategy_prefix]['strat_ret'] = (self.monthly_assessment[strategy.strategy_prefix]['strat_payoff'] - self.monthly_assessment[strategy.strategy_prefix]['strat_stake']) / self.monthly_assessment[strategy.strategy_prefix]['strat_stake']
            
            self.monthly_assessment[strategy.strategy_prefix]['strat_stake_cumsum'] = self.monthly_assessment[strategy.strategy_prefix]['strat_stake'].cumsum()
            self.monthly_assessment[strategy.strategy_prefix]['strat_payoff_cumsum'] = self.monthly_assessment[strategy.strategy_prefix]['strat_payoff'].cumsum()        
            self.monthly_assessment[strategy.strategy_prefix]['strat_cumret'] = (self.monthly_assessment[strategy.strategy_prefix]['strat_payoff_cumsum'] - self.monthly_assessment[strategy.strategy_prefix]['strat_stake_cumsum']) / self.monthly_assessment[strategy.strategy_prefix]['strat_stake_cumsum']
            self.monthly_assessment[strategy.strategy_prefix] = self.monthly_assessment[strategy.strategy_prefix][['as_of', 'strat_stake', 'strat_payoff', 'strat_ret', 'strat_stake_cumsum', 'strat_payoff_cumsum', 'strat_cumret']]

            self.monthly_assessment[strategy.strategy_prefix].loc[-1] = [self.monthly_assessment[strategy.strategy_prefix]['as_of'][0] + pd.offsets.MonthBegin(-1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # adding a row
            self.monthly_assessment[strategy.strategy_prefix].index = self.monthly_assessment[strategy.strategy_prefix].index + 1 # shifting index
            self.monthly_assessment[strategy.strategy_prefix] = self.monthly_assessment[strategy.strategy_prefix].sort_index() # sorting by index
    
    def plot_cumulative_return(self):
        if not self.assessment_has_been_performed:
            self.perform_assessment()
            
        plt.rcParams["figure.figsize"] = (14,12)

        # gca stands for 'get current axis'
        ax = plt.gca()

        i = 0
        for strategy in self.strategy_list:
            self.monthly_assessment[strategy.strategy_prefix].plot(kind='line', x='as_of', y='strat_cumret', marker=WageringStrategyAssessment.LINED_MARKERS[i][-1], color=WageringStrategyAssessment.LINED_MARKERS[i][0], label=strategy.strategy_prefix, lw=2, ax=ax)
            i += 1
        
        ax.axline([0, 0], [1, 1])

        plt.xlabel("Monthly", fontsize=15)
        plt.ylabel("Cumulative reurn to prior month", fontsize=15)
        plt.grid()
        plt.legend()

        plt.show()