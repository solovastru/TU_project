import numpy as np
from topsis_module import topsis_method
from ahp_module import ahp_weights
from inference import InferenceBN
from pgmpy.models import DiscreteBayesianNetwork


maintenance_actions = {
    "Roll_surface_damage": ["Check the roll surface for foreign objects contacting the roll surface.",
                            "Remove and regrind the roll to restore a smooth, uniform surface.",
                            "Replace the roll."],
    "Hydraulic_leaks": ["Refill hydraulic fluid.",
                        "Locate the leak visually or with leak detection tools and replace the defective part.",
                        "Replace the hydraulic pump."],

    "Blocked_cooling_channels": ["Inspect and clean suction strainers/filters.",
                                 "Replace cooling pipe or circuit.",
                                 "Replace the cooling pump."]

}

model = DiscreteBayesianNetwork.load('bn_tu_project.bif', filetype='bif') 
variables = ['Roll_surface_damage','Hydraulic_leaks', 'Blocked_cooling_channels']
obs_evidence = {'Pressure': "High"}

result_cause = InferenceBN(model, variables, obs_evidence)
result_cause = result_cause.most_probable()

actions = maintenance_actions[result_cause]


oee_importance= {("A", "A"): 1, ("A", "B"): 1/5, ("A", "C"): 1/6, 
                 ("B", "A"): 5, ("B", "B"): 1, ("B", "C"): 1,
                  ("C", "A"): 6, ("C", "B"): 1, ("C", "C"): 1}

spc_importance= {("A", "A"): 1, ("A", "B"): 3, ("A", "C"): 9, 
                 ("B", "A"): 1/3, ("B", "B"): 1, ("B", "C"): 6,
                  ("C", "A"): 1/9, ("C", "B"): 1/6, ("C", "C"): 1}

oee_spc_importance = {("oee", "oee"): 1, ("oee", "spc"): 2,
           ("spc", "oee"): 1/2, ("spc", "spc"): 1}



oee_alternatives_weights = ahp_weights(oee_importance)
spc_alternatives_weights = ahp_weights(spc_importance)
oee_spc_weights = ahp_weights(oee_spc_importance)



alternatives = list(key for key in oee_alternatives_weights.keys())
print(alternatives)


evaluation_matrix = np.array([
    [oee_alternatives_weights[action], spc_alternatives_weights[action]]
    for action in alternatives
])

#print(evaluation_matrix)
criteria_weights = [0.667, 0.333]



criterion =  ['max', 'max']


result = topsis_method(evaluation_matrix, criteria_weights, criterion)

mapped_results = dict(zip(actions, result))

best_action = max(mapped_results, key=mapped_results.get)
best_score = mapped_results[best_action]

print(f"The alarm was triggered by {result_cause}. The recommended maintenance action \"{best_action}\" has the best result when it comes to both OEE and spare part costs with a score of: \"{best_score:.3f}\"")
