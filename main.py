import networkx as nx
import random

import gurobipy as gp
from gurobipy import GRB

from itertools import combinations

from timeit import Timer
import csv

#Outputs a directed graph based on the filepath given as input. k sets the number of commodiites, r gives the range for the randomly chosen arc risks, costs and cost increases
def readingInput(path, k=5, r=25):
    assert isinstance(path, str), "readingTest-Error: path given is not a string"
    random.seed(r)
    G = nx.DiGraph()
    with open(path, "r") as reader:
        s = reader.readline()
# Skip Metadata
        while not("END OF METADATA" in s):
            s = reader.readline()
#Skip further lines by finding the first one that contains "1" (graph needs at least one outgoing edge from first node) and does not start with "~" which denotes comments
        while ("1" not in s) or (s[0] == "~"):
            s = reader.readline()
#Every line is converted to an arc, substract 1 from node numbers to switch from 1...n to 0...n-1, randomly set arc values (previously rcapacity for edge risk, length for edge cost, freeFlowTime for edge cost increase)
        while not( s == ""):
            numbers = s.split()
            edgeTail = int(numbers[0])-1
            edgeHead = int(numbers[1])-1
            #edgeRisk = float(numbers[2])
            #edgeCost = float(numbers[3])
            #edgeCostIncrease = float(numbers[4])
            edgeRisk = random.randint(1,r)
            edgeCost = random.randint(1, r)
            edgeCostIncrease = random.randint(1, r)
            G.add_edge(edgeTail, edgeHead, risk=edgeRisk, normalCost=edgeCost, costIncrease=edgeCostIncrease)
            s = reader.readline()
#Add random but valid origin-destination pairs
    random.seed(5)
    n = G.number_of_nodes()
    origDestPairs = {}
    for i in range(k):
        potProblem = True
        while potProblem:
            orig_i = random.randint(0, n - 1)
            dest_i = random.randint(0, n - 1)
            dests = origDestPairs.get(orig_i, [])
            potProblem = False
            if orig_i == dest_i: potProblem = True
            if not (nx.has_path(G, orig_i, dest_i)): potProblem = True
            if dest_i in dests: potProblem = True
        dests.append(dest_i)
        origDestPairs[orig_i] = dests
    nx.set_node_attributes(G, values=origDestPairs, name="destinations")
    return G


#Generate a random directed graph with n nodes and an average node degree of d. Place k commodities. Arc attributes set between 1 and r
def RandomGraphGen(n=15, d=6, r=25, k=5):
    G = nx.erdos_renyi_graph(n, d/n, seed=2, directed=1)
    randomRisk = {}
    randomNormalCost = {}
    randomCostIncrease = {}
    random.seed(1)
    for e in G.edges:
        randomRisk[e] = random.randint(1,r)
        randomNormalCost[e] = random.randint(1, r)
        randomCostIncrease[e] = random.randint(1, r)
    nx.set_edge_attributes(G, values=randomRisk, name="risk")
    nx.set_edge_attributes(G, values=randomNormalCost, name="normalCost")
    nx.set_edge_attributes(G, values=randomCostIncrease, name="costIncrease")
#    displayGraph(G)
    origDestPairs = {}
    for i in range(k):
        potProblem = True
        while potProblem:
            orig_i =  random.randint(0, n-1)
            dest_i = random.randint(0, n - 1)
            dests = origDestPairs.get(orig_i, [])
            potProblem = False
            if orig_i == dest_i: potProblem = True
            if not (nx.has_path(G,orig_i,dest_i)): potProblem = True
            if dest_i in dests: potProblem = True
        dests.append(dest_i)
        origDestPairs[orig_i] = dests
    nx.set_node_attributes(G, values=origDestPairs, name="destinations")
    return G


#Solves the MIP-formulation of the uncertainty-subproblem for a given graph. Utilizes custom callback.
def MultiCommodityMip(graph1, uncertaintyBudget=5):
#Check if input is a (directed) graph
    assert isinstance(graph1, (nx.classes.DiGraph, nx.classes.Graph)), "Not a DiGraph or Graph"
    if(type(graph1)==nx.classes.Graph): graph1 = nx.DiGraph(graph1)
# Ensure at least one edge exists for every required attribute
    assert nx.get_edge_attributes(graph1, "risk"), "No edge has a risk attribute"
    assert nx.get_edge_attributes(graph1, "normalCost"), "No edge has a normalCost attribute"
    assert nx.get_edge_attributes(graph1, "costIncrease"), "No edge has a costIncrease attribute"
    assert nx.get_node_attributes(graph1,"destinations"), "No node has a destinations attribute"
# Initialize list of origin-destination-pairs, for flow-constraint: If dest-orig-edge does not exist, add it and set its cost to practically infinite
    dests = nx.get_node_attributes(graph1,"destinations")
    odList = []
    for n in graph1.nodes():
        if n in dests:
            for d in dests[n]:
                odList.append((n,d))
    assert odList, "Number of origin-destination pairs is zero"
    tooHigh = graph1.number_of_nodes() * max(nx.get_edge_attributes(graph1, "normalCost").values())
    for (k1,k2) in odList:
        if (k2,k1) not in graph1.edges:
            graph1.add_edge(k2,k1, normalCost=tooHigh)
#Initialize model and variables
    m1 = gp.Model("mip1")
    m1._graph1 = graph1
    m1._odList = odList
    m1._y = m1.addVars([(e,k) for e in graph1.edges for k in odList], vtype=GRB.BINARY, name="y")
    m1._gamma = m1.addVars(graph1.edges, vtype=GRB.BINARY, name="gamma")
    m1._pi = m1.addVars([(e,k) for e in graph1.edges for k in odList], vtype=GRB.BINARY, name="pi")
    m1.update()
#Add objective function
    edgeRisks = nx.get_edge_attributes(graph1, "risk")
    opt = gp.LinExpr(0.0)
    for (e1,e2) in graph1.edges:
        riskCoeff = edgeRisks.get((e1,e2), 0)
        for (k1,k2) in odList:
            if (e1,e2) != (k2,k1): opt.add(m1._y[(e1,e2),(k1, k2)], riskCoeff)
    m1.setObjective(opt, GRB.MAXIMIZE)
    m1.update()
#Add flow constraints
    for (k1,k2) in odList:
        m1.addConstr(m1._y[(k2,k1),(k1,k2)] == 1, f"FlowCom{(k1, k2)}")
        for v in graph1.nodes:
            vBalance = gp.LinExpr()
            for (_,w) in graph1.out_edges(v):
                vBalance.add(m1._y[(v,w),(k1,k2)],1)
            for(w,_) in graph1.in_edges(v):
                vBalance.add(m1._y[(w,v),(k1,k2)], -1)
            m1.addConstr(vBalance==0, f"FlowBalanceNode{v}Com{(k1, k2)}")
    m1.update()
#Limit number of edges with increased cost
    uncCounter = gp.LinExpr()
    for e in graph1.edges:
        uncCounter.add(m1._gamma[e], 1)
    m1.addConstr(uncCounter <= uncertaintyBudget, "UncBudget")
    m1.update()
#Set pi-variables via pi(a,k)=1 <=> (y[a,k]=1 and gamma[a]=1)
    for k in odList:
        for e in graph1.edges:
            m1.addConstr(m1._pi[e,k] <= m1._y[e,k], f"Pi1Edge{e}Com{k}")
            m1.addConstr(m1._pi[e,k] <= m1._gamma[e], f"Pi2Edge{e}Com{k}")
            m1.addConstr(m1._pi[e,k] >= m1._y[e,k] + m1._gamma[e] -1, f"Pi3Edge{e}Com{k}")
    m1.update()
#Ensure no path has lower cost than the one indicated by the y-variables (find conflicting paths via callback)
    m1.Params.lazyConstraints = 1
    m1.optimize(callback=MultiCommodityCallback)
    return m1


#Callback used in the MultiCommodityMipFromGraph-method
def MultiCommodityCallback(model, where):
    if where == GRB.Callback.MIPSOL:
        assert isinstance(model._graph1, nx.classes.DiGraph), "Callback Error: model attribute is not a graph"
        normalCosts = nx.get_edge_attributes(model._graph1, "normalCost")
        costIncreases = nx.get_edge_attributes(model._graph1,"costIncrease")
        gamma = model.cbGetSolution(model._gamma)
# Initialize edge weights/costs based on variables of current MIP-solution
        weights = {}
        for e in model._graph1.edges:
            if gamma[e]:
                weights[e] = normalCosts.get(e,0) + costIncreases.get(e,0)
            else:
                weights[e] = normalCosts.get(e,0)
        nx.set_edge_attributes(model._graph1, values=weights, name="weight")
# For each commodity, compute cost of shortest path and cost of the chosen path (based on y- and pi)
        y = model.cbGetSolution(model._y)
        pi = model.cbGetSolution(model._pi)
        for (k1,k2) in model._odList:
            minCost = nx.shortest_path_length(model._graph1, source=k1, target=k2, weight="weight")
            currentCost = 0
            for e in model._graph1.edges:
                if y[e,(k1,k2)] and e != (k2,k1) and (e in normalCosts): currentCost += normalCosts[e]
                if pi[e,(k1,k2)] and e != (k2,k1) and (e in costIncreases): currentCost += costIncreases[e]
# If there are paths shorter than the chosen path, find one of them and add corresponding inequality
            if currentCost > minCost:
                ineqLeft = gp.LinExpr()
                for e in model._graph1.edges:
                    costCoeff = normalCosts.get(e, 0)
                    if e != (k2,k1): ineqLeft.add(model._y[e,(k1,k2)], costCoeff)
                    increaseCoeff = costIncreases.get(e, 0)
                    ineqLeft.add(model._pi[e,(k1,k2)], increaseCoeff)
                ineqRight = gp.LinExpr()
                path = nx.shortest_path(model._graph1, source=k1, target=k2, weight="weight")
                for i in range(len(path)-1):
                    u = path[i]
                    v = path[i+1]
                    ineqRight.add(normalCosts.get((u,v),0))
                    increaseCoeff = costIncreases.get((u, v), 0)
                    ineqRight.add(model._gamma[(u,v)], increaseCoeff)
                model.cbLazy(ineqLeft <= ineqRight)


#Returns a set of reduced graphs, where leaderBudget many arcs of g are eliminated. Only eliminates arcs used in the subproblem for g. Repeats random choice numReturn times.
def GraphReduceRandomly(g, leaderBudget=3, numReturn=1):
    random.seed(1)
    assert isinstance(g, (nx.classes.DiGraph, nx.classes.Graph)), "Not a DiGraph or Graph"
    m1 = MultiCommodityMip(g)
    y = []
    for (u,v) in g.edges:
        used = False
        for (o,d) in m1._odList:
            if m1._y[(u,v),(o,d)] and (u,v)!=(d,o): used = True
        if used: y.append((u,v))
    #y.remove((m1._destNode, m1._origNode))
    #numDeleted = int(len(y) * leaderBudget)
    GReturn = []
    for n in range(numReturn):
        gNew = nx.DiGraph(g)
        for i in range(leaderBudget):
            checkDuplicate = True
            while checkDuplicate:
                j = random.randint(0,len(y))
                if y[j] in gNew.edges:
                    gNew.remove_edge(*y[j])
                    checkDuplicate = False
        GReturn.append(gNew)
    return GReturn


#Similar to GraphReduceRandomly, except only the edgeConsidered number of edges with highest risk are eliminated. Set contains every possible choice of 3 arcs out of the 5 available.
def GraphReduceMostRisky(g, leaderBudget=3, edgesConsidered=5):
    GReturn = []
    m = MultiCommodityMip(g,0)
    y = m._y
    r = nx.get_edge_attributes(g, "risk", 0)
    odList = m._odList
#Filter for actually used edges and calculate their contribution to the risk-objective
    edgeRiskProduced = {}
    for (e1,e2) in g.edges:
        riskProduced = 0
        for (o,d) in odList:
            x = m.getAttr("X", [y[(e1,e2),(o,d)]])[0]
            if x and (e1,e2)!=(d,o): riskProduced += r[(e1,e2)]
        if riskProduced>0: edgeRiskProduced[(e1,e2)] = riskProduced
#Filter for the edges with highest risk-contribution
    innerChoice = []
    for i in range(edgesConsidered):
        if edgeRiskProduced:
            e = max(edgeRiskProduced, key=edgeRiskProduced.get)
            innerChoice.append(e)
            edgeRiskProduced.pop(e)
#Filter for all subsets with leaderBudget many edges and create corresponding reduced Graphs
    for j in range(leaderBudget):
        allCombs = combinations(innerChoice, j+1)
        for a in allCombs:
            G0 = nx.DiGraph(g)
            G0.remove_edges_from(a)
            GReturn.append(G0)
    return GReturn


#Measures the execution time of the MIP and writes it in a .csv-file. Graphs have up to 10*t nodes
def RandomGraphResults(t=10):
    lines = []
    for i in range(t):
        j = (i+1)*10
        t = Timer(f"MultiCommodityMip(g)", f"from __main__ import MultiCommodityMip; from __main__ import RandomGraphGen; g = RandomGraphGen({j})")
        g = RandomGraphGen(j)
        m = MultiCommodityMip(g)
        s = m.ObjVal
        lines.append([j, (min(t.repeat(repeat=2, number=5))), s])
        header = ["node number", "time", "ObjValue"]
        with open("results/RandomGraphScaling.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(lines)

def CityGraphResults(strings = ["inputFiles/SiouxFalls_net.tntp", "inputFiles/EMA_net.tntp", "inputFiles/friedrichshain-center_net.tntp", "inputFiles/berlin-prenzlauerberg-center_net.tntp", "inputFiles/Barcelona_net.tntp"]):
    lines = []
    for i in range(len(strings)):
        p = strings[i]
        t = Timer(f"MultiCommodityMip(g)", f"from __main__ import MultiCommodityMip; from __main__ import readingInput; g = readingInput(\"{p}\")")
        g = readingInput(p)
        nodeNumber = g.number_of_nodes()
        m = MultiCommodityMip(g)
        s = m.ObjVal
        lines.append([nodeNumber, (min(t.repeat(repeat=2, number=5))), s])
        header = ["node number", "time", "ObjValue"]
        with open("results/CityGraphScaling.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(lines)

#Analyze computation time and objective value (if feasible) of graphs returned by GraphReduceRandomly. input gives list of inputs for analyzed method
def ReduceRandmomlyResults(input = [1,4,8,10,12,16,20,25,30,35]):
    lines = []
    for i in input:
        t1 = Timer(f"G = GraphReduceRandomly(g,3,{i})",f"from __main__ import "
                                                                                          f"MultiCommodityMip; from "
                                                                                          f"__main__ import "
                                                                                          f"readingInput; from "
                                                                                          f"__main__ import "
                                                                                          f"GraphReduceRandomly;  g = "
                                                                                          f"readingInput("
                                                                                          f"\"inputFiles/berlin-prenzlauerberg-center_net.tntp\")")
        t2 = Timer(f"for g1 in G: MultiCommodityMip(g1)", f"from __main__ import "
                                                                                            f"MultiCommodityMip; from "
                                                                                            f"__main__ import "
                                                                                            f"readingInput; from "
                                                                                            f"__main__ import "
                                                                                            f"GraphReduceRandomly;  g = "
                                                                                            f"readingInput("
                                                                                            f"\"inputFiles/berlin-prenzlauerberg-center_net.tntp\"); G = GraphReduceRandomly(g,3,{i})")
        g = readingInput("inputFiles/berlin-prenzlauerberg-center_net.tntp")
        G = GraphReduceRandomly(g,3,i)
        obj = []
        for g1 in G:
            m = MultiCommodityMip(g1)
            if m.Status == GRB.Status.OPTIMAL:
                s = m.ObjVal
                obj.append(s)
        lines.append([i, (min(t1.repeat(repeat=1,number=1))+min(t2.repeat(repeat=1,number=1))), (min(obj))])
        header = ["leaderBudget", "time", "ObjValue"]
        with open("results/ReduceRandomlyResults.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(lines)


#Analyze computation time and objective value (if feasible) of graphs returned by GraphReduceMostRisky. input gives list of inputs for analyzed method
def ReduceMostRiskyResults(input = [3,4,5,6,7,8,9]):
    lines = []
    for i in input:
        t1 = Timer(f"G = GraphReduceMostRisky(g,3,{i})",f"from __main__ import "
                                                                                          f"MultiCommodityMip; from "
                                                                                          f"__main__ import "
                                                                                          f"readingInput; from "
                                                                                          f"__main__ import "
                                                                                          f"GraphReduceMostRisky;  g = "
                                                                                          f"readingInput("
                                                                                          f"\"inputFiles/berlin-prenzlauerberg-center_net.tntp\")")
        t2 = Timer(f"for g1 in G: MultiCommodityMip(g1)", f"from __main__ import "
                                                                                            f"MultiCommodityMip; from "
                                                                                            f"__main__ import "
                                                                                            f"readingInput; from "
                                                                                            f"__main__ import "
                                                                                            f"GraphReduceMostRisky;  g = "
                                                                                            f"readingInput("
                                                                                            f"\"inputFiles/berlin-prenzlauerberg-center_net.tntp\"); G = GraphReduceMostRisky(g,3,{i})")
        g = readingInput("inputFiles/berlin-prenzlauerberg-center_net.tntp")
        G = GraphReduceMostRisky(g,3,i)
        obj = []
        for g1 in G:
            m = MultiCommodityMip(g1)
            if m.Status == GRB.Status.OPTIMAL:
                s = m.ObjVal
                obj.append(s)
        lines.append([i, (min(t1.repeat(repeat=1,number=1))+min(t2.repeat(repeat=1,number=1))), (min(obj))])
        header = ["leaderBudget", "time", "ObjValue"]
        with open("results/ReduceMostRiskyResults.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(lines)


#Analyze the computation time and objective value of graph for different uncertainty budgets. inputs gives set of budgets to be tested
def UncScalingResults(inputs = [0,1,2,3,4,5,6,7,8,9,10,11,12]):
    lines = []
    for i in inputs:
        t = Timer(f"MultiCommodityMip(g,{i})", f"from __main__ import readingInput; from __main__ import MultiCommodityMip; g = readingInput(\"inputFiles/berlin-prenzlauerberg-center_net.tntp\")")
        g = readingInput("inputFiles/berlin-prenzlauerberg-center_net.tntp")
        m = MultiCommodityMip(g,i)
        obj = m.ObjVal
        lines.append([i, min(t.repeat(repeat=1,number=1)),obj])
        header = ["uncBudget", "time", "ObjValue"]
        with open("results/UncScalingResults.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(lines)


#Analyze the computation time and objective value of graph for different amounts of commodities. inputs gives commodity numbers to be tested
def ComScalingResults(inputs = [1,3,5,7,9,12,14,17,20,25,30]):
    lines = []
    for i in inputs:
        t = Timer(f"MultiCommodityMip(g)", f"from __main__ import readingInput; from __main__ import MultiCommodityMip; g = readingInput(\"inputFiles/berlin-prenzlauerberg-center_net.tntp\",k={i})")
        g = readingInput("inputFiles/berlin-prenzlauerberg-center_net.tntp",i)
        m = MultiCommodityMip(g)
        obj = m.ObjVal
        lines.append([i, min(t.repeat(repeat=1,number=1)),obj])
        header = ["comNumber", "time", "ObjValue"]
        with open("results/ComScalingResults.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(lines)

#Analyze the computation time and objective value of graph for different ranges for the randomly set arc risks, arc costs and arc cost increases. inputs gives ranges to be tested
def RandomRangeScalingResults(inputs = [1,5,10,15,20,25,30,35,40,45,50,75]):
    lines = []
    for i in inputs:
        t = Timer(f"MultiCommodityMip(g)", f"from __main__ import readingInput; from __main__ import MultiCommodityMip; g = readingInput(\"inputFiles/berlin-prenzlauerberg-center_net.tntp\",r={i})")
        g = readingInput("inputFiles/berlin-prenzlauerberg-center_net.tntp",r=i)
        m = MultiCommodityMip(g)
        obj = m.ObjVal
        lines.append([i, min(t.repeat(repeat=1,number=1)),obj])
        header = ["randomRange", "time", "ObjValue"]
        with open("results/RandomRangeScalingResults.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(lines)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    RandomGraphResults()
    CityGraphResults()
    ComScalingResults()
    UncScalingResults()
    RandomRangeScalingResults()
    ReduceRandmomlyResults()
    ReduceMostRiskyResults()