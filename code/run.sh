# case1_fast_upwind_TVD.pdf
#python main.py -n 300 -s -0.05 -e 0.5 -C 0.05 -t 0.05,0.1 -m TVD,Upwind -case 1 -i U -o U -w 0.1,0.5 -y 1,1.015,3,4,-0.95,-0.85,-0.15,0.05,-0.15,0.05,0.97,1.17,0.97,1.17
# case1_slow_upwind_TVD.pdf
#python main.py -n 300 -s -0.3 -e 0.5 -C 0.05 -t 0.4,0.6 -m TVD,Upwind -case 1 -i U -o U --watch=-0.3,-0.15 -y 2,5.5,4,8,-0.05,1.2,0.2,0.41,0.2,0.41,1.07,1.13,1.07,1.13
# case3_fast_upwind_TVD.pdf
python main.py -n 300 -s 0 -e 0.75 -C 0.05 -t 0.05,0.1 -m TVD,Upwind -case 3 -i U -o U --watch=0.3,0.7 -y 0.97,1.01,2.3,3.4,-1.1,-0.7,-0.1,0.4,-0.1,0.4,0.5,1.1,0.5,1.1
# case3_slow_upwind_TVD.pdf
#python main.py -n 300 -s -0.3 -e 0.5 -C 0.05 -t 0.4,0.6 -m TVD,Upwind -case 3 -i U -o U --watch=-0.3,-0.15 -y 2,5.5,4,8,-0.05,1.2,0.2,0.41,0.2,0.41,1.07,1.13,1.07,1.13
