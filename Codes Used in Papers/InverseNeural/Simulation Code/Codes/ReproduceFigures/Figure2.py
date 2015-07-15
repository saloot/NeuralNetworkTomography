#=======================IMPORT THE NECESSARY LIBRARIES=========================
import subprocess
import time
import os
#==============================================================================


#================================INITIALIZATIONS===============================
os.chdir('..')
os_name = os.name
if os_name == 'posix':
    shell_flag = True
else:
    shell_flag = False
#==============================================================================


#================================PLOT THE FIGURE===============================
print 'Figure 2.a'
query_str = 'python Plot_Results.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -M 4 -f "B" -K N -O "2249,4498,6747,8996,11245,13494"'
#proc = subprocess.Popen(query_str,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
proc = subprocess.Popen(query_str,stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell_flag)
time.sleep(2)

print 'Figure 2.b'
query_str = 'python Plot_Results.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -M 8 -f "B" -K N -O "100,2780,5460,8140,10820,13500"'
proc = subprocess.Popen(query_str,stdout=subprocess.PIPE, stderr=subprocess.STDOUT,shell=shell_flag)
time.sleep(2)

print 'Figure 2.c'
query_str = 'python Plot_Results.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -M 3 -f "B" -K N -O "2249,4498,6747,8996,11245,13494"'
proc = subprocess.Popen(query_str,stdout=subprocess.PIPE, stderr=subprocess.STDOUT,shell=shell_flag)
#==============================================================================



