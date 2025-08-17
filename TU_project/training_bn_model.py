from pgmpy.estimators import BayesianEstimator
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.base import DAG
from networkx.drawing.nx_pydot import to_pydot
from pgmpy.models import DiscreteBayesianNetwork
from discretize_data import DataProcessor


file_path = "C:\\Users\\roxan\\Desktop\\TU_project\\tu_project_categorized.xlsx"
#read the excel fiel
df = pd.read_excel(file_path)

#define cause columns
cause_cols = ['Roll_surface_damage','Hydraulic_leaks', 'Blocked_cooling_channels']

#transform into categorical states 
for col in df:
    df[col] = pd.Categorical(df[col], ordered=False)


#define parents and child nodes
parent_map = {
    'Roll_surface_damage': ["Vibration"],
    'Hydraulic_leaks': ["Pressure"], 
    'Blocked_cooling_channels': ["Flow_Meter"]
}

#show in which direction it goes
known_directions = {
    (parent, child)
    for parent, children in parent_map.items()
    for child in children
}


#create a DAG
G = DAG()

#add the nodes and edges
G.add_nodes_from({node for edge in known_directions for node in edge})  # add all nodes
G.add_edges_from(known_directions)

edges = G.edges()
print(G.edges())



G = nx.DiGraph()
G.add_edges_from(edges)

# Draw the graph
"""plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightgreen", font_size=10, arrowsize=20, arrowstyle='-|>')
plt.title("Bayesian Network Structure")
plt.show()"""

model_struct = DiscreteBayesianNetwork(ebunch=G.edges())
#train the model
model_struct.fit(
    data=df,
    estimator=BayesianEstimator,
    prior_type="dirichlet",
    pseudo_counts=1
)


model_struct.check_model()

#save the model for later use
model_struct.save('bn_tu_project.bif', filetype='bif')