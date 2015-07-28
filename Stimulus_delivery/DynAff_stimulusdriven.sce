# --- Dynamic Affect: Stimulusdriven --- #
# 
# Creates a Presentation-logfile and custom logfile.
#
# Note: to convert edf file to asc type in windows cmd:
# edf2asc D:\USERS\Snoek\log\<file>.edf
#
# Lukas Snoek, Dynamic Affect project (15/16), 
# Research Master Psychology

# --------------- SDL headers --------------- #

scenario_type = fMRI;
#scenario_type = fMRI_emulation;
#scan_period = 2000;
pulses_per_scan = 1;
pulse_code = 100;

default_text_color = 255,255,255;	
default_background_color = 0,0,0;
default_font_size = 25;
default_font = "arial";

response_matching = simple_matching;
active_buttons = 3;
button_codes = 1,2,3;
response_logging = log_all;
no_logfile = false;   

begin; 

# --------------- SDL definitions: objects --------------- #

TEMPLATE "DynAff_SDL_prepost.tem";

# --------------- START OF PCL --------------- #
begin_pcl;

# Set trk_test in "global_variables.pcl"
include "DynAff_globalvariables.pcl";
include "DynAff_PCL_prepost.pcl";

output_file out = new output_file;
out.open (filename + "_DynAff_stimulusdriven_" + version + ".txt");

# Fix filename
string logname = logfile.filename();
string new_name = logname.replace(".log","") + "_" + version + ".log";
logfile.set_filename(new_name);

out.print("Cat"); 	out.print("\t");	
out.print("T_abs"); 	out.print("\t");
out.print("T_rel"); 	out.print("\t");
out.print("Val"); 	out.print("\n");

include "DynAff_functions.pcl";

# --------------- EYETRACKING STUFF ---------- #
# Some parameter for the eyetracker.

# Suffix for .edf filename
string filename_suffix = "s";
include "DynAff_tracker_init.pcl"

# --------------- (STIMULI) ARRAYS ----------- #

# Stimuli
array <string> stim_names[48] =
	{"2158.jpg","2311.jpg","2340.jpg","4623.jpg","8350.jpg","8461.jpg","8540.jpg","Faces_134_h.jpg",								 # situation/pos
	 "6313.jpg","9163.jpg","9254.jpg","9420.jpg","9427.jpg","9433.jpg","9921.jpg","Faces_145_v.jpg", 								 # situation/neg
	 "2026.jpg","2521.jpg","2593.jpg","2840.jpg","Faces_306_v.jpg","Faces_311_h.jpg","Faces_336_h.jpg","People_150_h.jpg",   # situation/neu
	 "char1_pos_cauc.jpg","char2_pos_cauc.jpg","char3_pos_cauc.jpg","char4_pos_cauc.jpg","char5_pos_marok.jpg","char6_pos_marok.jpg","char7_pos_marok.jpg","char8_pos_cauc.jpg", # faces/pos/cauc-marok
	 "char1_neg_cauc.jpg","char2_neg_cauc.jpg","char3_neg_cauc.jpg","char4_neg_cauc.jpg","char5_neg_marok.jpg","char6_neg_marok.jpg","char7_neg_marok.jpg","char8_neg_cauc.jpg", # faces/neg/cauc-marok
	 "char1_neu_cauc.jpg","char2_neu_cauc.jpg","char3_neu_cauc.jpg","char4_neu_cauc.jpg","char5_neu_marok.jpg","char6_neu_marok.jpg","char7_neu_marok.jpg","char8_neu_cauc.jpg"};# faces/neu/cauc-marok

array <int> stim_codes[48] =
	{11000,11000,11000,11000,11000,11000,11000,11000,		# situation/pos
	 21000,21000,21000,21000,21000,21000,21000,21000,		# situation/neg
	 31000,31000,31000,31000,31000,31000,31000,31000,		# situation/neu
	 12100,12100,12100,12100,12200,12200,12200,12100,		# faces/pos/cauc-marok
	 22100,22100,22100,22100,22200,22200,22200,22100,		# faces/pos/cauc-marok
	 32100,32100,32100,32100,32200,32200,32200,32100};		# faces/pos/cauc-marok

array <int> stim_idx[48] =
	{1,2,3,4,5,6,7,8,
	 9,10,11,12,13,14,15,16,
	 17,18,19,20,21,22,23,24,
	 25,26,27,28,29,30,31,32,
	 33,34,35,36,37,38,39,40,
	 41,42,43,44,45,46,47,48};
stim_idx.shuffle();

introtrial1.present();
introtrial2.present();

# --------------- WAIT FOR PULSE-trial ------------- #
pulsetrial.present();
int current_pulse = pulse_manager.main_pulse_count();	
loop until ( pulse_manager.main_pulse_count() > current_pulse ) begin end;

# timer = start of experiment
timer = clock.time();
tracker.send_message("Start");

# Write starting time
out.print("Start");			out.print("\t");
out.print(string(timer)); 	out.print("\t");
out.print(string(0));	 	out.print("\n");

# Wait 5 secs
wait_ISI(timer, 5000);

int j = 1;

loop until j > stim_idx.count() begin;
	
	# Draw evaluation
	go_eval = update_eval_counter(stim_idx[j]);
	
	# Prepare stim
	exp_bitmap.set_filename(stim_names[stim_idx[j]]);
	exp_bitmap.load();
	experimental_event.set_event_code(string(stim_codes[stim_idx[j]]+go_eval*1000000));
	
	# Start trial; set codes
	start_pic = clock.time();
	tracker.send_message(string(stim_codes[stim_idx[j]]+go_eval*1000000));
   
	experimental_trial.present();
	
	out.print(string(stim_idx[j])); 		out.print("\t");
	out.print(string(start_pic)); 		out.print("\t");
	out.print(string(start_pic-timer));	out.print("\t");
	
	# Update & wait
	j = j + 1;
	
	if go_eval == 0 then 
		out.print("\n"); 
	else		
		run_evaluation(0, 0, 1000, 1);	
	end;
	
	wait_ISI(start_pic, stim_dur + ISI);
	
end;

include "tracker_wrapup.pcl"

end_trial.present();
