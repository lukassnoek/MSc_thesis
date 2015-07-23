# --- Dynamic Affect: Pretest --- #
# Experimental script to run the Pretest scenario
# of the Dynamic Affect paradigm. Runs an event-related
# presentation of stimuli, which are in 50% of the trials
# (counterbalenced over conditions) followed by a
# likert-scale evaluation (very pos - very neg).
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
pulses_per_scan = 1000;
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
out.open (filename + "_DynAff_pretest_" + version + ".txt");

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
string filename_suffix = "a";
include "DynAff_tracker_init.pcl"

# Regular execution of intro trial
introtrial1.present();

# -------------- Practice-trial ------------- #
#exp_bitmap.set_filename("practice.jpg");
#exp_bitmap.load();

#int now = clock.time(); 
#wait_ISI(now, 2000);

#practice_trial.present();
#exp_bitmap.unload();

#run_evaluation(0, 0, 1000, 1);

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

# --------------- MAIN EXPERIMENTAL LOOP ------------- #
# Loops until all stimuli have been presented. 
# In 50% of the trials (counterbalanced over conditions),
# an evaluation-trial is presented.

loop int i = 1 until i > stimuli_condition.count() begin;

	stimulus_category = stimuli_condition[i]; # random condition is drawn
	box_position.shuffle(); # Random starting position is draw (as first index, later)
	
	exp_bitmap.unload();
	
	# Check condition, set stimulus, and update counter (i_xx)
	if (stimulus_category == 10) then
		exp_bitmap.set_filename(face_stim[1]);
		if face1_eval[i_f1] == 1 then go_eval = 1; else go_eval = 0; end;
		i_f1 = i_f1 + 1;
	elseif (stimulus_category == 11) then
		exp_bitmap.set_filename(face_stim[2]);
		if face2_eval[i_f2] == 1 then go_eval = 1; else go_eval = 0; end;
		i_f2 = i_f2 + 1;
	elseif (stimulus_category == 12) then
		exp_bitmap.set_filename(face_stim[3]);
		if face3_eval[i_f3] == 1 then go_eval = 1; else go_eval = 0; end;
		i_f3 = i_f3 + 1;
	elseif (stimulus_category == 20) then
		exp_bitmap.set_filename(house_stim[1]);
		if house1_eval[i_h1] == 1 then go_eval = 1; else go_eval = 0; end;
		i_h1 = i_h1 + 1;
	elseif (stimulus_category == 21) then
		exp_bitmap.set_filename(house_stim[2]);
		if house2_eval[i_h2] == 1 then go_eval = 1; else go_eval = 0; end;
		i_h2 = i_h2 + 1;
	elseif (stimulus_category == 22) then
		exp_bitmap.set_filename(house_stim[3]);
		if house3_eval[i_h3] == 1 then go_eval = 1; else go_eval = 0; end;
		i_h3 = i_h3 + 1;
	end;
	
	# Present experimental trial!
	exp_bitmap.load();
	start_pic = clock.time();
	experimental_event.set_event_code(string(stimulus_category+go_eval*100));
	tracker.send_message(string(stimulus_category+go_eval*100));
   
	experimental_trial.present();
	
	# Write to custom logfile
	out.print(string(stimulus_category)); out.print("\t");
	out.print(string(start_pic)); out.print("\t");
	out.print(string(start_pic-timer)); out.print("\t");
	
	if go_eval == 0 then 
		out.print("\n"); 
	else		
		run_evaluation(0, 0, 1000, 1);	
	end;
	
	wait_ISI(start_pic, stim_dur+ISI);
	
	i = i + 1;

end; 

include "tracker_wrapup.pcl"

end_trial.present();


