
from pgmpy.inference import BeliefPropagation, VariableElimination, CausalInference

#load the trained model
#fault_detection_model = DiscreteBayesianNetwork.load('bn_tu_project.bif', filetype='bif') 


class InferenceBN:
    def __init__(self, model, variables: list, observed_evidence):
        self.model = model
        self.variables = variables
        self.observed_evidence = observed_evidence
        self.inference = VariableElimination(model)



    def run_inference(self) -> dict:
            """
            Run inference for the given evidence and return probabilities.

            """
            results = {}
            for var in self.variables:
                result = self.inference.query(variables=[var], evidence=self.observed_evidence)
                prob_yes = result.values[1] 
                results[var] = prob_yes
            return results

    def most_probable(self) -> str:
        """
        Get the most probable variable given the evidence.
        """
        results = self.run_inference()
        return max(results, key=results.get)
