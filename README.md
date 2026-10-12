# cy530-network-security-research-methods

##Summary##
This repository contains code, made using Claude AI, related to the CY530 Research Methods paper. The goal of this research is to test various machine learning models on network reconnaissance data, and examine how they perform in terms of accuracy and efficiency.
This research uses the CIC-IDS2017 dataset, more specifically: Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
This program was made using Claude AI, and was based on the following project: https://github.com/JONATHANJOHN95001/acn-network-intrusion-detection

##Environment##
This code was tested in a Linux Virtual Machine, using Debian with a desktop environment. Using VirtualBox, this VM was allocated 3 CPUs, 4 GB of RAM, and 30 GB of storage.
This code should be reproducible in any Linux environment, however, I have only tested it in this environment.

##Steps##
1. Download the dataset found here: https://www.unb.ca/cic/datasets/ids-2017.html
2. Download the referenced code repository found here: https://github.com/JONATHANJOHN95001/acn-network-intrusion-detection
3. Download this GitHub repository
4. Once everything is downloaded, move the python script from this repository into the /acn-network-intrusion-detection folder, and move the MachineLearningCSV folder, into the /acn-network-intrusion-detection folder as well. For the MachineLearningCSV folder, make sure that only the CSV files are in the folder (by default, the folder is nested such as MachineLearningCSV/MachineLearningCSV/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv). The folder should not be nested.
5. In a terminal inside the /acn-network-intrusion-detection directory, run the following: 
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
6. Use the following command to run the program:
python portscan_dt_rf_svm.py

##Results##


##Sources / AI Disclosure##
1. https://www.unb.ca/cic/datasets/ids-2017.html
2. https://github.com/JONATHANJOHN95001/acn-network-intrusion-detection
3. Claude AI was used to create the portscan_dt_rf_svm.py Python script. It was also used to debug and analyze the existing implementation.




