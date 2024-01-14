# This script sets up the fuzzy variables, membership functions, and rules.

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Step 1: Define Fuzzy Variables
positivity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'positivity')
negativity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'negativity')
sentiment = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'sentiment')

# Step 2: Membership Functions
# Positivity
positivity['low'] = fuzz.trimf(positivity.universe, [0, 0, 0.5])
positivity['high'] = fuzz.trimf(positivity.universe, [0.5, 1, 1])
# Negativity
negativity['low'] = fuzz.trimf(negativity.universe, [0, 0, 0.5])
negativity['high'] = fuzz.trimf(negativity.universe, [0.5, 1, 1])
# Sentiment Output
sentiment['negative'] = fuzz.trimf(sentiment.universe, [0, 0, 0.5])
sentiment['positive'] = fuzz.trimf(sentiment.universe, [0.5, 1, 1])

# Step 3: Fuzzy Rules
rule1 = ctrl.Rule(positivity['high'] & negativity['low'], sentiment['positive'])
rule2 = ctrl.Rule(positivity['low'] & negativity['high'], sentiment['negative'])

# Step 4: Control System Creation and Simulation
sentiment_analysis_ctrl = ctrl.ControlSystem([rule1, rule2])
sentiment_analysis = ctrl.ControlSystemSimulation(sentiment_analysis_ctrl)
