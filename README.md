# HallTensorAnalyze
Analyze the spin and charge conductivity tensor of your material under the framwork of magnetic point group and spin group

  
| Author | Jianting Dong |
| ------ | ------------- |
| Language | Python |
| Time | 2025-11-25 |
| Version | 2.3 |

## SOC Mode  

* usage:
```bash
python Analyze_Hall_tensor.py -f mpg_RuO2.txt > RuO2_soc.out
```
* The first three lines of `mpg_RuO2.txt` are lattice vector. 
* Other lines are the operations, with the formart of `number operation(no spacing)` 
* Operations are from the website of [Findsym](https://iso.byu.edu/findsym.php) 
* Example:  
```bash
4.533000000000 0.000000000000 0.000000000000
0.000000000000 4.533000000000 0.000000000000
0.000000000000 -0.000000000000 3.124000000000
1 x,y,z,+1
2 x,-y,-z,+1
3 -x,y,-z,+1
4 -x,-y,z,+1
5 -x,-y,-z,+1
6 -x,y,z,+1
7 x,-y,z,+1
8 x,y,-z,+1
9 -y,-x,-z,-1
10 -y,x,z,-1
11 y,-x,z,-1
12 y,x,-z,-1
13 y,x,z,-1
14 y,-x,-z,-1
15 -y,x,-z,-1
16 -y,-x,z,-1
```

## Non-SOC Mode  

* usage
```bash
python Analyze_Hall_tensor.py -f fsg_RuO2.cif.json -noso > RuO2_nosoc.out
```
* The `fsg_RuO2.cif.json` is from the [FindSpinGroup](https://app.findspingroup.com/) with the following steps:  
	1. Upload your `cif` file  
	2. Set the magnetic moments  
	3. Click the button `Click to download RAW data of output`  
