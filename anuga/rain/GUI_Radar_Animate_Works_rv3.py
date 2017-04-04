"""
BOM RADAR & RAIN GAUGE DATA PROCESSING 

This script will Read a directory tree of RADAR rainfall and show or save the grid image

Additional Options Include:
 - can also plot a reference polyline over the grid image
 - can Extract rainfall hyetograph from Radar data at Rain Gauge Locations
 - plot comparitive plot of Gauge Vs Radar

Note the original files are gzipped, and could use
gzip.open('myfile.gz') to read directly instead of first unzipping all files!!


import gzip

f_name = 'file.gz'

filtered = []
with gzip.open(f_name, 'r') as infile:
    for line in infile:
        for i in line.split(' '):
            if i.startswith('/bin/movie/tribune'):
                filtered.append(line)
                break # to avoid duplicates





A final addition is to extract rainfall time series at nominated gauge locations from RADAR data
gauge_id_list = ['570002','570021','570025','570028','570030','570032','570031','570035','570036',
                 '570047','570772','570781','570910','570903','570916','570931','570943','570965',
                 '570968','570983','570986','70214','70217','70349','70351']


DATA NEEDS:
- 1 Radar Base Directory to read radar files
- Radar Files Time Step
- 2 Radar_UTM_LOC_file 

- 3 file(s) of reference polylines to plot
- 4 Reference Rain Gauge location file


"""
import gzip
import os
import sys
import glob
import numpy as np
from scipy.io import netcdf
import pylab as pl
from easygui import *
import fnmatch
import matplotlib.pyplot as plt
from anuga.fit_interpolate.interpolate2d import interpolate2d


print 'START...'
cur_dir= os.getcwd()
top_dir = os.path.dirname(cur_dir)
print cur_dir
print top_dir
keep_running = True

MainMenulistChoices = ["1_Get_RADAR_Files_Base_DIR",
                   "2_Get_RADAR_UTM_LOC_FILE",
                   "3_Get_Reference_Poly_File(s)",
                   "4_Get_RainGauge_Location_FILE",


                   "A_Show_Radar_Plots_on_screen",
                   "B_Save_Radar_Plots_to_Image_File",
                   "C_Process file without images",
                   "D_Extract rainfall from RADAR at gauge LOC & Plot to Screen",
                   
                   "Z-EXIT"]
#================  DEFINE REQUIRED FUNCTIONS ===========================================================

# ============ DEFINE CLASS TO STORE PERSISTENT SETTINGS======================================
class Settings(EgStore):

    def __init__(self, filename):  # filename is required
        #-------------------------------------------------
        # DEFINE THE VARIABLES TO STORE
        #-------------------------------------------------
        self.Last_1RADAR_DIR = ""
        self.Last_2RADAR_UTM_Ref_File = ""
        self.Last_3REF_PolyList = ""
        self.Last_4RainGauge_Loc_File = ""
        

        #-------------------------------------------------
        # For subclasses of EgStore, these must be
        # the last two statements in  __init__
        #-------------------------------------------------
        self.filename = filename  # this is required
        self.restore()            # restore values from the storage file if possible
# ============ DEFINE CLASS TO STORE PERSISTENT SETTINGS======================================

# ----------------------------------------------------------------------------------------

def get_poly_2_plot_over_radar():
    # ------------------ POLY LINE PLOT OPTION -----------------------------------
    title = "ADD PolyLine to Plot"
    msg = "Select PolyLine File such as ACT Bdy"
    plot_line = ynbox(msg, title)
    
    plotmore = plot_line
    while plotmore:
        title = "Select a Polyline File to add to Plot (Coords relative to RADAR)"
        msg = "Select PolyLine eg: ACT Bdy, could be LL, UTM or Local"
        default = '01_ACT_State_Bdy_Relative_to RADAR.csv'
        plot_file = fileopenbox(msg, title, default)
        polylist = np.genfromtxt(plot_file,delimiter=",", dtype=(float, float)) # COuld be several poly's
        title = "ADD Another PolyLine to Plot"
        msg = "Select PolyLine File such as ACT Bdy"
        plotmore = ynbox(msg, title)
        print plotmore
        print'Polylist...'           
        print polylist
        #'data' is a matrix containing the columns and rows from the file
        xl = polylist[:,0]  # Python indices are (row,col) as in linalg
        yl = polylist[:,1]  # Creates arrays for first two columns
        print 'xl,yl...'
        print xl,yl
    return(plot_line,xl,yl)  # Poly Line X,Y list(s)

def get_raingauge_location_and_label():
    # ------------------ RAIN GAUGES TO EXTRACT RADAR DATA AT SELECTED POINTS ----------------------
    Gauge_LOC_points = []
    Gauge_LOC_Labels = []
    title = "Select Raingauges Location file to extract DATA from RADAR"
    msg = "Select Location File For Raingauge Extraction in LL?"
    extract_raingauge = ynbox(msg, title)
    if extract_raingauge :
        # Open Raingauge Location File
        title = "Select a Raingauge Location file to Extract RADAR data"
        msg = "Select File"
        default = '02_Rain_Gauge_Station_Location_subset.csv'
        RainGauge_LOC_file = fileopenbox(msg, title,default)
        fid = open(RainGauge_LOC_file)
        lines = fid.readlines()  # Read Entire Input File
        fid.close()         
        for line in lines:
            print line
            line=line.strip('\n')
            fields = line.split(',')
            Gauge_LOC_Labels.append(fields[0])
            Gauge_LOC_points.append([float(fields[1]),float(fields[2])]) # ---=== THIS IS THE RAIN GAUGE LOCATIONS
        print Gauge_LOC_points
    return(RainGauge_LOC_file,extract_raingauge,Gauge_LOC_points,Gauge_LOC_Labels)
    
def convert_radar_to_UTM():
    # ------------------ CONVERT RADAR FROM LAT LONG TO UTM --------------------------------
    # PROCESS THE Lat Long to Produce UTM Grid of Radar ??
    title = "Select RADAR UTM Reference File to Convert Lat Long to UTM"
    msg = "Select RADAR UTM Location File to Convert from Lat Long?"
    Convert2UTM = ynbox(msg, title)
    if Convert2UTM :
        # Open Radar UTM reference file
        title = "Select a RADAR UTM Location Reference File"
        msg = "Select File"
        Radar_UTM_LOC_file = fileopenbox(msg, title)
        fid = open(Radar_UTM_LOC_file)
        lines = fid.readlines()  # Read Entire Input File
        fid.close()         
        for line in lines:
            print line
            line=line.strip('\n')
            fields = line.split(',')
            offset_x = float(fields[0])
            offset_y = float(fields[1])
        print offset_x,offset_y
        #raw_input('Hold here... line 128')
        return(Convert2UTM,offset_x,offset_y)

def SET_Radar_File_TimeStep():
    # ------------------ SET RADAR DATA TIME STEP -----------------------------------
    # Could use file name details to automate this, but what happens if filename format changes??
    msg='Enter Time Step (minutes) expected in the Radar Rainfall files'
    title='SELECT Time Step in Minutes Check the RADAR Files..'
    default = 10
    Time_Step = integerbox(msg,title,default,lowerbound=1,upperbound=1440)
    return(Time_Step)




def Convert_LLPrecip2UTM(offset_x,offset_y,x,y,precip):
    # ------------- OPTION TO CONVERT TO UTM Implimentation FIRST  ====================================
    # Only saves to a file not used for plots YET !!
    if Convert2UTM :
        print 'Converting to UTM...'
        for i in x:
            for j in y:
                print x[i],y[j]
                UTM_x = x[i]*1000.0 + offset_x
                UTM_y = y[j]*1000.0 + offset_y  
                s = ' %.3f,%.3f,%.3f \n' %(UTM_x,UTM_y,precip[0][i])
                outfid.write(s)
                print UTM_x,UTM_y,precip[0][i]  # Need to reference the correct element in precip !!  CHECK THIS ....
        outfid.close()
    return()                   


def Extract_RADAR_Data_at_Gauge_Loc():
    # ------------- OPTION TO EXTRACT RAINFALL FROM RADAR GRID AT GAUGE POINTS FIRST  ====================================                    
    if extract_raingauge : 
        print 'Extract Rain Check data First'
        #print x
        #print y
        #print precip[0]
        x = data.variables['x_loc'][:]
        if y[0] < 0:
            y = data.variables['y_loc'][:]  # Check if y[0] = -ve if not reverse...  arr[::-1]
        else:
            y = data.variables['y_loc'][::-1]  # Check if y[0] = -ve if not reverse...  arr[::-1]
        Z = data.variables[precip_name][:]
        #print x
        #print y
        #print Z[0]
        print Gauge_LOC_points[0]
        # and then do the interpolation
        values = interpolate2d(x,y,Z,Gauge_LOC_points) # This is a numpy array of Rain for this time slice at each of the gauge locations
        values.tolist()      # Convert array to list
        print 'First values...'
        #print values
        ALL_values.append(values.tolist() ) # This is a time history of the List above
        print 'First ALL values...'
        #print ALL_values
        # Save it to a file...??
        #raw_input('Hold at Gauge Extract....line 278')
    return()


def Radar_Plot_Save_AndShow():
    # SHOW and SAVE the Plot            
    if processing == 3:
        
        plt.figure(1)
        plt.clf()
        if plot_line : plt.plot( xl, yl,'--w')
        plt.suptitle('RADAR RAINFALL'+filename[-20:-3], size=20)
        s_title = 'Max Rainfall = %.1f mm / period' % (np.max(precip))
        plt.title(s_title , size=12)
        plt.imshow(precip, origin='lower', interpolation='bicubic',extent=(x.min(), x.max(), y.min(), y.max()),vmin=0, vmax=Daily_plot_Vmax)
        plt.colorbar()            
        plt.show()
        plt.savefig('RADAR RAINFALL'+filename[-20:-3]+'.jpg',format='jpg')
    return()



def Radar_Plot_Save_NoShow():
    # DONT SHOW only SAVE the Plot
    if processing == 2:
        print 'Saving Image...'   
        plt.figure(1)
        plt.clf()
        if plot_line : plt.plot( xl, yl,'--w')
        plt.suptitle('RADAR RAINFALL'+filename[-20:-3], size=20)
        s_title = 'Max Rainfall = %.1f mm / period' % (np.max(precip))
        plt.title(s_title , size=12)
        plt.imshow(precip, origin='lower', interpolation='bicubic',extent=(x.min(), x.max(), y.min(), y.max()),vmin=0, vmax=Daily_plot_Vmax)
        plt.colorbar()            
        plt.savefig('RADAR RAINFALL'+filename[-20:-3]+'.jpg',format='jpg')
    return()

def Radar_Plot_ShowOnly():
    # ONLY SHOW the PLOT DONT SAVE
    if processing == 1:
        plt.figure(1)
        plt.clf()
        if plot_line : plt.plot( xl, yl,'--w')
        plt.suptitle('RADAR RAINFALL'+filename[-20:-3], size=20)
        s_title = 'Max Rainfall = %.1f mm / period' % (np.max(precip))
        plt.title(s_title , size=12)
        plt.imshow(precip, origin='lower', interpolation='bicubic',extent=(x.min(), x.max(), y.min(), y.max()),vmin=0, vmax=Daily_plot_Vmax)
        plt.colorbar()            
        plt.show()
    return()

def Plot_Time_Hist_Radar_at_Gauge_Loc():
    #+++++++++++  NOW PLOT THE DATA EXTRACTED AT GAUGE LOCATIONS ++++++++++++++++
    if processing == 4:
        gauge_count = 0
        t = np.arange(File_Counter)
        for item in sorted(Gauge_LOC_Labels):
            # Access List of Lists [rows][cols]
            # need to plot gauge in columns
            #      loc1 = [[0.0 for y in range(ncols)] for x in range(nrows)]
            #bar_value = zip(*ALL_values)
            bar_values = [x[gauge_count] for x in ALL_values]
            total_rain = sum(bar_values)
            Ave_rain = total_rain/len(bar_values)
            max_Intensity = max(bar_values)/Time_Step*60.0
            b_title = 'Tot rain = %.1f mm/hr, Ave. rain = %.1f mm, Max Int.= %.1f mm' % (total_rain,Ave_rain,max_Intensity)
            #   Using list comprehension  b=[x[0] for x in a]
            print len(t)
            print len(bar_values)
            # Think about using...  zip(*lst)
            plt.bar(t,bar_values)
            plt.suptitle(' Rain Gauge data for Station %s:' % item, fontsize=14, fontweight='bold')    
            plt.title(b_title)
    
            plt.xlabel('time steps')
            plt.ylabel('rainfall (mm)')
            plt.show()
            gauge_count+=1
    return()


def Plot_Radar_Accumulated():
    print precip_total
    print 'maximum rain =',np.max(precip_total)
    print 'mean rain =',np.mean(precip_total)
    Total_Rain_Vol = np.mean(precip_total)/1000.0*128.0*128.0 # Volume in Million m3 over 128km x 128km area
    # using a = np.array   can get  np.min(a), np.max(a) and np.mean(a)
    Peak_Intensity = Rain_Max_in_period/Time_Step*60
    print 'Total rainfall volume in Mill m3 =',Total_Rain_Vol
    print 'Peak rainfall in 1 time step = ', Rain_Max_in_period
    print 'Peak Intensity in 1 timestep =',Peak_Intensity
    dir_part = os.path.basename(os.path.normpath(fromdir))
    plt.figure(1)
    plt.clf()
    plt.suptitle('Accumulated '+str(File_Counter)+' files '+Plot_SupTitle, size=20)
    
    
    s_title = 'Max Int. = %.1f mm/hr, Ave. rain = %.1f mm, Tot rain Vol. = %.3f Mill. m3' % (Peak_Intensity,np.mean(precip_total),Total_Rain_Vol)
    plt.title(s_title , size=12)
    
    plt.imshow(precip_total, origin='lower', interpolation='bicubic',extent=(x.min(), x.max(), y.min(), y.max()))
    if plot_line : plt.plot( xl, yl,'--w')
    plt.colorbar()
    plt.show()
    return()


# =============================== MAIN LINE CODE =============================================    

print 'Start INTERACTIVE ROUTINES....'
Keep_Processing = True
#-----------------------------------------------------------------------
#  DEFINE SETTINGS FILE and RESTORE DEFINED VARIABLES FOR USE
#-----------------------------------------------------------------------
settingsFile = "01_Radar_animate_V3_Settings.txt"
settings = Settings(settingsFile)
settings.restore() # Read from the Settings File...

extract_raingauge = False
Daily_plot_Vmax = 15.0
# From here a directory tree is read, then depending on option flagged by
#   - Save Radar Plots
#   -

while Keep_Processing: # ===== OPTION TO PROCESS MORE RADAR DATA ======================================
    # Check Last used files ???
    settings.restore()

    Last_1RADAR_DIR = settings.Last_1RADAR_DIR
    Last_2RADAR_UTM_Ref_File = settings.Last_2RADAR_UTM_Ref_File
    Last_3REF_PolyList = settings.Last_3REF_PolyList
    Last_4RainGauge_Loc_File = settings.Last_4RainGauge_Loc_File
    
    mess1 = 'Last RADAR DIR Used: '+Last_1RADAR_DIR +'\nLast RADAR UTM File Used: '+Last_2RADAR_UTM_Ref_File +'\nLast REF PolyList File: '+Last_3REF_PolyList+'\n'
    mess2 = 'Last Rain Gauge file: '+Last_4RainGauge_Loc_File
    message = mess1+mess2
    reply = choicebox(message,"SELECT MENU ITEM...", MainMenulistChoices) # THIS IS MAIN MENU !!!!

    ###  ---   PROCESS THE OPTIONS ---- #############################################################
    if "1_Get_RADAR_Files_Base_DIR" in reply:
        print 'Present Directory Open...'
        title = "Select Directory to Read Multiple rainfall .nc files"
        msg = "This is a test of the diropenbox.\n\nPick the directory that you wish to open."
        Last_1RADAR_DIR = diropenbox(msg, title)
        
        fromdir = Last_1RADAR_DIR
        rootPath = fromdir
        pattern = '*.nc'
        #pattern = '*.gz'
    elif "2_Get_RADAR_UTM_LOC_FILE" in reply:
        Convert2UTM,offset_x,offset_y = convert_radar_to_UTM()
        
    elif "3_Get_Reference_Poly_File(s)" in reply:
        # Get Plot Flag and Polyline to plot
        plot_line,xl,yl = get_poly_2_plot_over_radar()
    elif "4_Get_RainGauge_Location_FILE" in reply:
        RainGauge_LOC_file,extract_raingauge,Gauge_LOC_points,Gauge_LOC_Labels = get_raingauge_location_and_label()    
        Last_4RainGauge_Loc_File = RainGauge_LOC_file
    elif "A_Show_Radar_Plots_on_screen" in reply:
        # GET DATA and PLOT
        Radar_Plot_ShowOnly()
        Plot_Radar_Accumulated()
    elif "B_Save_Radar_Plots_to_Image_File" in reply:
        Radar_Plot_Save_NoShow()
    elif "C_Process file without images" in reply:
        # GET DATA and PROCESS
        pass
    elif "D_Extract rainfall from RADAR at gauge LOC & Plot to Screen" in reply:
        # Extract Rainfall at Gauge Locations from RADAR GRid
        Extract_RADAR_Data_at_Gauge_Loc()
        Plot_Time_Hist_Radar_at_Gauge_Loc()
        Plot_Radar_Accumulated()
                   
                
    print rootPath
    First = True
    File_Counter = 0
    Rain_Max_in_period = 0.0
    if extract_raingauge : ALL_values =[]
    
    # ======================= DIRECTORY OPEN LOOP ======================================
    # LOOP Through directories to process RADAR Rainfall and Accumulate Total
    for root, dirs, files in os.walk(rootPath): 
        
        for filename in fnmatch.filter(files, pattern): 
            
            print 'Number of Files = ',len(fnmatch.filter(files, pattern))
            #print( os.path.join(root, filename))
            #print root,dirs
            print filename
            print 'PART Filename...'
            print filename[-20:-3]
            #raw_input('Hold here... line 179')
            if Convert2UTM :
                outfilename = filename[0:-4]+'.xyz' # Output file for RADAR in UTM
                outfid = open(outfilename, 'w')
            # Create a file for each time slice... or 1 file for ALL??
            """
            if extract_raingauge:
                ext_outfilename = filename[0:-4]+'_Ext_Raon.xyz' # Output file for EXTRACTING RADAR Rain at Gauge
                extoutfid = open(ext_outfilename, 'w')
            """
            File_Counter +=1
            if File_Counter == 1 and processing <> 4:
                msg = "This will be the title for the PLOTS...."
                title = "Enter Title Text"
                default = "RADAR_Data_"+filename[-20:-3]
                strip = True
                Plot_SupTitle = enterbox(msg, title,default,strip)
    
            # Now read NetCDF file and Plot the Radar Rainfall Array
            """
            data = NetCDFFile(filename, netcdf_mode_r)
            print 'VARIABLES:'
            print data.variables        
            The default format for BOM data is Lat Lon,
            
            """
            if pattern == '*.gz':
                #gzip.open(filename)
                filename = gzip.open(filename, 'rb')
                print filename
                data = netcdf.NetCDFFile(os.path.join(root,filename), 'r') # RADAR NetCDF files have Dimensions, Attributes, Variables
            else:
                data = netcdf.NetCDFFile(os.path.join(root, filename), 'r') # RADAR NetCDF files have Dimensions, Attributes, Variables
            print 'VARIABLES:'
            #print data.variables  
            #print data.__dict__
            print 'Reference LAT, LONG = ',data.reference_longitude, data.reference_latitude
            #print 'ATTRIBUTES:'
            #print data.attributes              
            #raw_input('Hold here... line 217')
            possible_precip_names = ['precipitation',  'precip', 'rain_amount']  # This handles format changes in the files from BOM !!!!
            # Go through each of the possible names
            for name in possible_precip_names:  # Check if name is a key in the variables dictionary
                if name in data.variables:
                    precip_name = name
                    print 'BOM Reference name tag in this file:'
                    print precip_name

# --- END for name -----------------------------------
            
            if First:
                First = False
                try:
                    precip = data.variables[precip_name].data # The BOM files use precipitation, precip, and rain_amount ???
                    #print data.variables['precipitation'].data
                    precip_total = precip.copy() # Put into new Accumulating ARRRAY
                    print ' Accumulate rainfall here....'
                    x = data.variables['x_loc'].data
                    y = data.variables['y_loc'].data
                    Rain_Max_in_period  = max (np.max(precip),Rain_Max_in_period)  
                    


                    """
                    # My original attempt...
                    for current_gauge in Gauge_LOC_points:
                        print current_gauge
                        values = interpolate2d(x,y,precip[0], current_gauge)  
                        print values
                    raw_input('Hold here...')
                    """
                    
                    """
                    # Steves Original Attempt
                    x = data.variables['x_loc'][:]
                    y = data.variables['y_loc'][:]
                    UTM_x = x*1000.0 + offset_x
                    UTM_y= y*1000.0 + offset_y
                    Z = data.variables['precip'][:]
                    # and then do the interpolation
                    values = interpolate2d(x,y,Z,points)
                    """                        
                        
                #except:
                except Exception,e:  # mIGHT NOT NEED THIS ANT MORE ...???
                    print str(e)


                    precip = data.variables['precip'].data # precip = data.variables['rain_amount'].data
                    precip_total = precip.copy()
                    print ' Keep accumulating rainfall....'
                    
                    x = data.variables['x_loc'][:]
                    y = data.variables['y_loc'][:]
                    UTM_x = x*1000.0 + offset_x
                    UTM_y= y*1000.0 + offset_y
                    Z = data.variables['precip'][:]
                    # and then do the interpolation
                    #values = interpolate2d(x,y,Z,points)  #  CHECK WHAT THIS IS DOING !!!!!
                    
                    
                    #x = data.variables['x_loc'].data
                    #y = data.variables['y_loc'].data        
                    #UTM_x = x*1000.0 + offset_x
                    #UTM_y = y*1000.0 + offset_y  
                    Rain_Max_in_period  = max (np.max(precip),Rain_Max_in_period)  

                    if extract_raingauge :    # ----- CHECK IF THIS IS NEEDED HERE  ????
                        print 'Extract Rain in Except'
                        #values = interpolate2d(x,y,precip, Gauge_LOC_points)                      



            else:  # ---If NOT FIRST !!!
                try:
                    precip = data.variables[precip_name].data
                    #print data.variables['precipitation'].data
                    precip_total += precip
                    print ' Keep accumulating rainfall....'
                    x = data.variables['x_loc'].data
                    y = data.variables['y_loc'].data
                    Rain_Max_in_period  = max (np.max(precip),Rain_Max_in_period)
                    # CONVERT TO UTM ???
                    
                    
                    # ------------- OPTION TO EXTRACT RAINFALL FROM RADAR GRID AT GAUGE POINTS FIRST  ====================================                    
                    if extract_raingauge : 
                        print 'Extract Rain At Gauges from RADAR......'
                        #print x
                        #print y
                        #print precip[0]
                        x = data.variables['x_loc'][:]
                        if y[0] < 0:
                            y = data.variables['y_loc'][:]  # Check if y[0] = -ve if not reverse...  arr[::-1]
                        else:
                            y = data.variables['y_loc'][::-1]  # Check if y[0] = -ve if not reverse...  arr[::-1]
                        Z = data.variables[precip_name][:]
                        #print x
                        #print y
                        #print Z[0]
                        print Gauge_LOC_points[0]
                        # and then do the interpolation
                        values = interpolate2d(x,y,Z,Gauge_LOC_points) # This is a numpy Array.... change to List ??
                        values.tolist()
                        #np.array([[1,2,3],[4,5,6]]).tolist()
                        print 'Values....'
                        #print values
                        ALL_values.append(values.tolist())
                        print 'ALL Values....'
                        #print ALL_values
                        #raw_input('Hold at Gauge Extract....line 373')
                        
                           
                except:
                    precip = data.variables['precip'].data # precip = data.variables['precip'].data
                    precip_total += precip
                    print ' Keep accumulating rainfall....'
                    x = data.variables['x_loc'].data
                    y = data.variables['y_loc'].data
                    Rain_Max_in_period  = max (np.max(precip),Rain_Max_in_period)
                    if extract_raingauge : 
                        print 'Extract Rain in Except'
                        #values = interpolate2d(x,y,precip, Gauge_LOC_points)   

                                            
            #raw_input('hold..')
            #print x
            # ----================    SHOW the PLOT and SAVE   =====================--------------------

            if processing == 0:    
                pass
        #---==== END  For Filename ======================--------------------------
        if extract_raingauge :
            print ALL_values
        #raw_input('Hold at Gauge Extract....line 454')
        
    # ---====={{{{  END for dir }}}}}}=======-------------------
 
    title = "Run AGAIN"
    msg = "Select another DIR?"
    keep_running = ynbox(msg, title)
# NOTE in Windows USE PHOTO LAPSE to create an AVI ....
# Then MEDIACODER to convert to MPG to reduce size !

# In Ubuntu Use: 
#  mencoder mf://*.jpg -mf w=800:h=600:fps=5:type=jpg -ovc lavc \ > -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o output.avi
#  mencoder mf://*.jpg -mf w=800:h=600:fps=5:type=jpg -ovc lavc     -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o output.avi


