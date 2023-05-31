#python main.py -n 1000 -s -1 -e 1 -C 0.05 -t 0 -m LaxWendroff -case 1 -i U -o U
#python main.py -n 200 -s -1 -e 1 -C 0.1 -t 0,0.2 -m LaxWendroff -case 1 -i U -o U
python main.py -n 200 -s -1 -e 1 -C 0.1 -t 0,0.2,0.4,0.6 -m Upwind -case 1 -i U -o U
#python main.py -n 100 -s -3 -e 1 -C 0.1 -t 0,0.05,0.1 -m LaxWendroff -case 0 -i U -o U
#python main.py -n 70 -s -8 -e 8 -C 0.02 -t 0,0.8,1.6 -m LaxWendroff -case 3 -i U -o W
#python main.py -n 1000 -s -8 -e 8 -C 0.02 -t 0,0.8,1.6 -m LaxWendroff -case 3 -i U -o U
#python main.py -n 261 -s 0 -e 1 -C 0.1 -t 1.6 -m LaxWendroff -case 3
#python main.py -n 261 -s 0 -e 1 -C 0.1 -t 0 -m LaxWendroff -case 3
