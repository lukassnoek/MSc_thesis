# --- Dynamic Affect: NARRATIVE --- #
# Experimental script to run the Narrative-scenario
# of the Dynamic Affect paradigm. Runs an presentation 
# of stimuli with fixed onsets, calibrated to the narrative,
# which are called "nar-stimuli". In between passages of 
# the narrative, the same stimuli are presented in isolation
# (called "iso-stimuli"), in a fixed order of blocks of 6.
# There are 6 different versions of this script, with each
# a different order of this block.
#
# Also, in two occasions, multiple stimuli are presented at 
# the same time (which was a b*tch to program...), called
# "special-trials" (a variety of nar-trials). 
#
# Creates a Presentation-logfile and custom logfile.
#
# Note: to convert edf file to asc, type in windows cmd:
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
default_font_size = 20;
default_font = "arial";
default_text_align = align_left;

response_matching = simple_matching;
active_buttons = 3;
button_codes = 1,2,3;
response_logging = log_all;
no_logfile = false;   
default_monitor_sounds = false;

begin; 

# --------------- SDL definitions: objects --------------- #

TEMPLATE "DynAff_SDL_narrative.tem";

#--------------- BEGIN PCL --------------------#

begin_pcl; 

# Set trk_test in "global_variables.pcl"
include "DynAff_globalvariables.pcl";

include "DynAff_PCL_narrative.pcl";

# --------------- LOGFILE stuff -------------- #
getOutput.present();
string filename = system_keyboard.get_input(askMe,text2);

# Ask for version
getOutput.present();
string version = system_keyboard.get_input(askMe2,text3);

output_file out = new output_file;
out.open (filename + "_DynAff_narrative_" + version + ".txt");

out.print("Type");			out.print("\t");
out.print("Cat");				out.print("\t");
out.print("T_abs");			out.print("\t");
out.print("T_rel");			out.print("\n");
	
out.print("Start");			out.print("\t");
out.print(string(100));		out.print("\t");

# Get indices for stimulus locations
int idx_char_jan = int(version.substring(3,1)); 
int idx_char_mat = int(version.substring(4,1)); 
int idx_char_ben = int(version.substring(5,1));
int idx_loc_jan = int(version.substring(6,1)); 
int idx_loc_mat = int(version.substring(7,1));
int idx_loc_ben = int(version.substring(8,1));

# Recode according to version
face1_bm.set_filename(jpg_char[idx_char_jan]); face1_bm.load();
face2_bm.set_filename(jpg_char[idx_char_mat]); face2_bm.load();
face3_bm.set_filename(jpg_char[idx_char_ben]); face3_bm.load();
house1_bm.set_filename(jpg_loc[idx_loc_jan]);  house1_bm.load();
house2_bm.set_filename(jpg_loc[idx_loc_mat]);  house2_bm.load();
house3_bm.set_filename(jpg_loc[idx_loc_ben]);  house3_bm.load();

# Fix filename
string logname = logfile.filename();
string new_name = logname.replace(".log","") + "_" + version + ".log";
logfile.set_filename(new_name);

# --------------- EYETRACKING STUFF ---------- #
# Some parameter for the eyetracker.

# Suffix for .edf filename
string filename_suffix = "n";
include "DynAff_tracker_init.pcl"

# Start off with introduction trial
introduction_trial.present();
introduction_picture.add_part(intro_part2, 0, 150);
introduction_trial.present();
introduction_picture.add_part(intro_part3, 0, 0);
introduction_trial.present();
introduction_picture.add_part(intro_part4, 0, -150);
introduction_trial.present();

audiotest_pic.present();
int current_pulse = pulse_manager.main_pulse_count();	
loop until ( pulse_manager.main_pulse_count() > current_pulse ) begin end;

audiotest.present();

int now = clock.time();
loop until clock.time() > (now+10000) begin end; 

introduction_trial2.present();

# Wait for scanner (until pulse = 2)
pulsetrial.present();
current_pulse = pulse_manager.main_pulse_count();	
loop until ( pulse_manager.main_pulse_count() > current_pulse ) begin end;

int timer = clock.time();

# Start default screen (black) and audio file!
default.present();
narrative.present();

# Print onset of narrative to logfile
out.print(string(timer));			out.print("\t");
out.print(string(timer-timer));	out.print("\n");

# While loop until all stimuli have passed
loop until stop == 1 begin;
	
	if (iso_onset[i_iso] > nar_onset[i_nar]) then
		go_iso = 0;
		current = "nar";
	end;
	
	# If there is a special narrative trial, then:
	if ((i_nar) == nar_special[i_spec]) && current == "nar" then
		
		# Pick narrative stimulus
		stimulus_category = nar_stim[i_nar];
		actual_onset = nar_onset[i_nar];
		
		int it = 1;
		int start_special = clock.time();
		
		# Loop to present three stimuli next to each other
		loop until it > 3 begin
			
			# Add appropriate stimulus according to stimulus_category
			if (stimulus_category == 10) then 
				exp_pic.add_part(face1_bm, loc_special[it], 0);
				exp_pic.add_part(jan_cap, loc_special[it], 450);
				
			elseif (stimulus_category == 11) then 
				exp_pic.add_part(face2_bm, loc_special[it], 0);
				exp_pic.add_part(mat_cap, loc_special[it], 450);

			elseif (stimulus_category == 12) then 
				exp_pic.add_part(face3_bm, loc_special[it], 0);
				exp_pic.add_part(ben_cap, loc_special[it], 450);

			elseif (stimulus_category == 20) then 
				exp_pic.add_part(house1_bm, loc_special[it], 0);	
				exp_pic.add_part(locjan_cap, loc_special[it], 450);

			elseif (stimulus_category == 21) then 
				exp_pic.add_part(house2_bm, loc_special[it], 0);
				exp_pic.add_part(locmat_cap, loc_special[it], 450);

			elseif (stimulus_category == 22) then 
				exp_pic.add_part(house3_bm, loc_special[it], 0);	
				exp_pic.add_part(locben_cap, loc_special[it], 450);

			end;
			
			# Set to the right location
			exp_event.set_event_code(string(stimulus_category+1000));
			
			# Wait until next presentation
			loop until (clock.time() - timer) > actual_onset begin end;
			tracker.send_message(string(stimulus_category+1000)); 
			exp_trial.present();
			
			out.print(current);									out.print("\t");
			out.print(string(stimulus_category+1000));	out.print("\t");
			out.print(string(clock.time()));					out.print("\t");
			out.print(string(clock.time()-timer));			out.print("\n");				
			
			if (i_nar < 3 || i_nar == 10 || i_nar == 11) then
				actual_onset = nar_onset[i_nar+1];
			else
				int stop_now = clock.time();
				loop until (clock.time() > stop_now + ISI) begin end;
			end;
			
			i_nar = i_nar + 1;
			
			stimulus_category = nar_stim[i_nar];
			actual_onset = nar_onset[i_nar];
			
			it = it + 1;
			
		end;	
		
		exp_pic.clear();
		default.present();
		
		if i_spec < 2 then
			i_spec = i_spec + 1;
		end;
		
	end;		
	
	# Check whether iso or nar is first:
	if (iso_onset[i_iso] < nar_onset[i_nar]) then
		go_iso = 1;
	else
		if i_nar == nar_onset.count() then
			go_iso = 1;
		end;
	end;
	
	if go_iso == 1 then
		
		# If iso is the next one, pick a random stim
		int idx = i_iso % 6;
		if idx == 0 then 
			idx = 6; 
			iso_stim.shuffle();
		end;
		
		stimulus_category = iso_stim[idx];
		actual_onset = iso_onset[i_iso];
		
		i_iso = i_iso + 1;
		
		current = "iso";
		
	else
		# If nar is first, check fixed stimulus order and set stim accordingly
		stimulus_category = nar_stim[i_nar];
		actual_onset = nar_onset[i_nar];
		
		if i_nar < nar_stim.count() then
			i_nar = i_nar + 1;
		end;
		
		current = "nar";
		go_iso = 0;
	end;
	
	# Set stimulus according to stim_cat
	if (stimulus_category == 10) then exp_pic.add_part(face1_bm, 0, 0);
		elseif (stimulus_category == 11) then exp_pic.add_part(face2_bm, 0, 0);	
		elseif (stimulus_category == 12) then exp_pic.add_part(face3_bm, 0, 0);
		elseif (stimulus_category == 20) then exp_pic.add_part(house1_bm, 0, 0);	
		elseif (stimulus_category == 21) then exp_pic.add_part(house2_bm, 0, 0);
		elseif (stimulus_category == 22) then exp_pic.add_part(house3_bm, 0, 0);	
	end;
	
	# Show default screen until onset
	if current == "iso" then
		actual_onset = actual_onset + delay;
	elseif current == "nar" then
		stimulus_category = stimulus_category + 100; # to disentangle from iso_stim
	end;
	
	loop until (clock.time() - timer) > actual_onset begin end;
	start_pic = clock.time();
	
	# Run experimental trial until stim_duration has ended
	exp_event.set_event_code(string(stimulus_category));
	tracker.send_message(string(stimulus_category)); 
	exp_trial.present();
		
	out.print(current); 								out.print("\t");
	out.print(string(stimulus_category));		out.print("\t");
	out.print(string(clock.time()));				out.print("\t");
	out.print(string(clock.time()-timer));		out.print("\n");						

	loop until clock.time() > (start_pic + stim_dur) begin end;
	
	# Remove stimulus from exp_event
	exp_pic.remove_part(1);
	
	# Got back to default screen
	default.present();
	
	if i_iso == (iso_onset.count()+1) then stop = 1; end; 
	
end;

include "tracker_wrapup.pcl"

end_trial.present();
