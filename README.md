
## The Comparison results of different NP metahuristic algo for VRPTW

Running Algorithms on dataset: rc108.txt

- - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - -

Best-Known Solution (BKS) Route Cost: 1114.2

BKS solution:

Route #1: 2 6 7 8 46 4 45 5 3 1 100

Route #2: 12 14 47 17 16 15 13 9 11 10

Route #3: 33 32 30 28 26 27 29 31 34 93

Route #4: 41 42 44 43 40 38 37 35 36 39

Route #5: 61 81 94 71 72 54 96

Route #6: 64 51 76 89 18 48 19 20 66

Route #7: 65 83 57 24 22 49 21 23 25 77

Route #8: 69 98 88 53 78 73 79 60 55 70 68

Route #9: 82 99 52 86 87 59 97 75 58 74

Route #10: 90

Route #11: 92 95 67 62 50 63 85 84 56 91 80

- - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - -

Hybrid Genetic Search (HGS) Route Cost: 1114.2

HGS solution:

Route #1: 12 14 47 17 16 15 13 9 11 10 

Route #2: 82 99 52 86 87 59 97 75 58 74 

Route #3: 65 83 57 24 22 49 21 23 25 77 

Route #4: 64 51 76 89 18 48 19 20 66 

Route #5: 92 95 67 62 50 63 85 84 56 91 80 

Route #6: 33 32 30 28 26 27 29 31 34 93 

Route #7: 61 81 94 71 72 54 96 

Route #8: 41 42 44 43 40 38 37 35 36 39 

Route #9: 2 6 7 8 46 4 45 5 3 1 100 

Route #10: 90 

Route #11: 69 98 88 53 78 73 79 60 55 70 68 

- - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - -

Guided Local Search (GLS) Route Cost: 1266.9

GLS solution:

Route #1: 71 72 44 43 40 38 37 35 36 39

Route #2: 98 82 90 53 78 73 79 2 60

Route #3: 92 67 32 30 28 26 27 29 31 34 93

Route #4: 65 99 24 22 20 49 21 23 25 77

Route #5: 95 51 76 89 33 50 62 91 80

Route #6: 12 14 47 17 16 15 13 9 11 10

Route #7: 88 6 7 8 46 4 45 5 3 1 100 55

Route #8: 69 70 61 81 94 96 54 41 42 68

Route #9: 83 52 57 86 87 59 97 75 58 74

Route #10: 64 19 48 18 63 85 84 56 66

- - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - -

Ant Colony Optimization (ACO) Route Cost: 1321.8459204561746

ACO solution:

Route #1: 69 98 88 82 99 52 86 74 57 83 66 91

Route #2: 65 64 51 76 89 85 63 62 56 80

Route #3: 90 53 73 79 78 60 55 68

Route #4: 33 28 30 32 34 31 29 27 26

Route #5: 72 71 93 94 81 61 54 96 100 70

Route #6: 2 45 5 8 7 6 46 4 3 1

Route #7: 41 42 44 38 39 40 36 35 37 43

Route #8: 19 21 23 18 48 49 22 20 24 25

Route #9: 12 14 47 17 16 15 11 10 9 13

Route #10: 59 58 87 97 75 77

Route #11: 92 95 84 50 67

- - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - -

Simulated Annealing (SA) Route Cost: 1237.620141359753

SA solution:

Route #1: 7 8 46 4 45 5 3 1 100 55

Route #2: 64 51 76 89 63 85 84 56 91

Route #3: 69 98 53 12 15 16 17 47 14

Route #4: 90 82 9 13 11 10

Route #5: 61 42 44 40 39 38 37 35 36 43

Route #6: 65 52 86 77 25 23 57

Route #7: 88 60 78 73 79 6 2 70 68

Route #8: 92 67 62 34 50 94 96

Route #9: 99 87 59 97 75 58 74

Route #10: 83 24 22 19 18 48 21 49 20 66

Route #11: 81 93 71 72 41 54

Route #12: 95 33 32 30 28 26 27 29 31 80

- - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - -

Algorithms   No. of Routes  Costs    Gap(%)     Runtime(seconds)

  BKS	          11	        1114.2	  -	-

  HGS	          11	        1114.2	  0.0	      300.137736082077

  GLS	          10	        1266.9	  13.7	    300.0492959022522

  ACO	          11	        1321.8	  18.63	    877.1980187892914

  SA	          12	        1237.6	  11.08	    416.80780506134033

- - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - -

![3 HGS rc108](https://github.com/user-attachments/assets/afac9be9-37a1-4b4d-a7f8-5d346d653e2f) ![3 GLS rc108](https://github.com/user-attachments/assets/b8140b7f-30c2-4f65-9d55-24207754dd0f) ![3 ACO rc108](https://github.com/user-attachments/assets/3643d0e9-c3d4-418c-8ed7-efea8d94ede9) ![3 SA rc108](https://github.com/user-attachments/assets/3632158b-6131-4157-97bd-9cd4116bcecf)




## VRPTW-Solver-Comparison-pyApp
Solving Vehicle Routing Problems with Time Windows using multiple Classic Heuristic and metaheuristic algorithms (Hybrid Genetic Search (HGS), Guided Local Search (GLS), Ant Colony Optimization (ACO), Simulated Annealing (SA)), then comparing results with each one's results and present it in the graph. This was my master's thesis project.

## keywords
Vehicle Routing Problems with Time Window (VRPTW), Hybrid Genetic Search (HGS), Guided Local Search (GLS), Ant Colony Optimization (ACO), Simulated Annealing (SA), MACS-VRPTW, Genetic Algorithm (GA), Exact, Heuristics, Metaheuristics, Machine Learning Algorithms, GRASP, Local Search (LS), Neighborhood Search, OR-Tools, VRP, VRPTW, Vehicle Routing Problems (VRP), pyVRP.

## Important Note
Using the Python Jupyter Notebook is highly advised. While compiling, each model executes independently. However, if you run the code straight from main.py , the application may crash and display errors due to the ACO procedure’s use of multiple threads! However, it functions perfectly now that I’m using the Python Jupyter Notebook code.

## More Details
To have a thorough understanding (about the parameter tuning and machanise) of my Vehicle Routing Problem with Time Windows project, I recommend reading my master's thesis. Thank you.

# Development

## Create venv

```sh
python -m venv .venv
```

## Activate venv

```sh
. .venv/bin/activate
```

## Install requirements

```sh
pip install -r requirements.txt
```
## Run the Project

```sh
python main.py
```

## Clone git repository URL 

```sh
git clone {paste repository URL}
```
