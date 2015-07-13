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

#scenario_type = fMRI;
scenario_type = fMRI_emulation;
scan_period = 2000;
pulses_per_scan = 300;
pulse_code = 30;

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

# Introduction text (example)
text {
	caption = 

	"Nu volgt instructie over de taak voor dit blok. Druk steeds op
	een van de knoppen onder je rechterhand om door te gaan.";
	
	font_size = 25; 
} intro_part1;

text {
	caption = 
	
	"Luister goed naar het verhaal en kijk goed naar de afbeeldingen die 
	op het scherm komen. De afbeeldingen kunnen op twee manieren voorkomen: \n";

} intro_part2;

text {
	caption = 
	
	"1. Tijdens het verhaal. Hier correspondeert de afbeelding met 
	de persoon of locatie die op dat moment in het verhaal voorkomt.
	Probeer zo snel mogelijk de afbeeldingen aan de karakters en locaties 
	in het verhaal te koppelen en te onthouden! \n";

} intro_part3;				

text {
	caption = 
	
	"2. Tussen passages in het verhaal. De afbeelding wordt getoond los van 
	het verhaal. Welke stimulus wanneer voorkomt, is volledig willekeurig!
	Probeer bij deze afbeeldingen te bedenken bij welke personages/locaties
	ze horen! \n";
	
} intro_part4;		
		
# Introduction text2
text {
	caption = 

	"Probeer je goed in te leven in het verhaal zometeen. \n
				  
	Ook is het belangrijk dat je zo min mogelijk beweegt,
	je ogen openhoudt, en niet in slaap valt. \n \n
				
	Als je er klaar voor bent,
   druk dan op een knop om te beginnen!"; 
	
	font_size = 30;
	text_align = align_center;
} introduction2;

# Narrative audio
wavefile { filename = "Narrative_final_36.wav";} narrative_file;

# Experimental bitmap
bitmap {	filename = "char_janos.jpg";} face1_bm;			# Janos
bitmap {	filename = "char_matthias.jpg";} face2_bm;			# Benji
bitmap {	filename = "char_benji.jpg";} face3_bm;			# Matthias
bitmap {	filename = "loc_janos.jpg";} house1_bm;	# House Janos
bitmap {	filename = "loc_matthias.jpg";} house2_bm; # House Benji 
bitmap {	filename = "loc_benji.jpg";} house3_bm; # House Matthias

# End
text {
			caption = 	"Dit is het einde van dit deel van het experiment."; 
			font =		"arial";		
			font_size = 25;
} end;

# --------------- SDL definitions: PICTURES/SOUNDS --------------- #

# Intro
picture {text intro_part1; x = 0; y = 300;} introduction_picture;
picture {text introduction2; x = 0; y = 0;} introduction_picture2;

# Narrative
sound { wavefile narrative_file;
		  } narrative;

# End
picture {text end; x = 0; y = 0;} end_picture;

# Pulse
picture {text { caption = "We gaan bijna beginnen. \n Even wachten op de scanner..."; 
					 font="arial"; font_size = 20;}; 
					 x=0; y=0;} pulsetrial;

# Experimental trial
picture {background_color = 0,0,0;} exp_pic;

# Background
picture {background_color = 0,0,0;} default;

# a picture for text input (getFilename)
picture {
	text { 
		caption = "Enter the output filename:"; font_color=255,255,255; };
		x = 0; y = 100;
		text { 
			caption = " "; } text2;
		x = 0; y = 0;
} askMe;

picture {background_color = 128,128,128;} et_calibration;

# --------------- SDL definitions: TRIALS --------------- #

# Intro trial
trial {
	all_responses		= true; 
	trial_type 			= first_response;
	trial_duration 	= forever;
		stimulus_event {
			picture introduction_picture;
			time = 0;
		} introduction_event;
} introduction_trial;

# Intro trial
trial {
	all_responses		= true; 
	trial_type 			= first_response;
	trial_duration 	= forever;
		stimulus_event {
			picture introduction_picture2;
			time = 0;
		} introduction_event2;
} introduction_trial2;

# End trial
trial {
	trial_duration = 5000; 
	stimulus_event {
		picture end_picture;
	} end_event;
} end_trial;

trial {
   picture askMe;
} getOutput;

trial {
	
	stimulus_event {
		picture exp_pic;
		code = 0;
	} exp_event;
} exp_trial;

#--------------- BEGIN PCL --------------------#

begin_pcl; 

# --------------- EYETRACKING STUFF ---------- #
# Some parameter for the eyetracker.

# just testing? (1 = yes)
int trk_test = 1; 

# Suffix for .edf filename
string filename_suffix = "n";
include "DA_tracker_init.pcl"

# --------------- (STIMULI) ARRAYS ----------- #
# Array with isolation stimuli
array <int> iso_stim[6] =
	{10,11,12,20,21,22};

# Array with narrative stimuli
array <int> nar_stim[48] =
	{10,11,12,10,11,12,11,12,
	 10,20,21,22,20,21,22,10,
	 21,11,22,12,20,21,20,22,
	 11,20,11,20,22,12,10,22,
	 21,21,12,11,20,10,22,21,
	 10,12,20,22,11,12,21,10};

# Array with onset times for isolation stimuli
array <int> iso_onset[36] = 
{49750,85650,143590,167900,213800,
247700,322560,367700,414080,466900,
521270,567400,611280,650770,672820,
706670,732820,767180,805130,854350,
906660,946150,968720,1015900,1051790,
1092310,1125130,1157440,1212310,1252310,
1288210,1336920,1378970,1422560,1460000,
1484620};

# Array with onset times for narrative stimuli
array <int> nar_onset[48] =
{3900,4800,5500,21700,68300,111400,175900,206660,
221500,237950,238970,240000,266540,287180,300000,309000,
333130,376290,423500,448000,477160,509400,529550,548800,
576350,584610,682560,717000,725641,751092,774871,796923,
813023,848734,897109,914850,983697,1023590,1045228,1061500,
1083589,1100000,1108703,1144876,1179000,1221538,1238301,1298846};

# Indices for "special" nar_stim, when succeeding stimuli should be juxtaposed
array <int> nar_special[2] =
	{1,10};
	
# Locations for special cases:
array <int> loc_special[3] =
	{-500, 0, 500};

# --------------- PCL parameters ------------- #
int delay = 1000; # before stim starts
int stim_dur = 2000; # stimulus duration
int ISI = 4000; # inter-stimulus interval

# Iterables
int i_iso = 1; 
int i_nar = 1;
int i_spec = 1;

# Stop-condition
int stop = 0;

# Declaration of other stuff
int stimulus_category = 0;
int actual_onset = 0;
int start_pic = 0;
string current = "";
int go_iso;

# --------------- LOGFILE stuff -------------- #
getOutput.present();
string filename = system_keyboard.get_input(askMe,text2);

output_file out = new output_file;
out.open (filename + "_narrative.txt");

out.print("Type");			out.print("\t");
out.print("Cat");				out.print("\t");
out.print("T_abs");			out.print("\t");
out.print("T_rel");			out.print("\n");
	
out.print("Start");			out.print("\t");
out.print(string(0));		out.print("\t");

# Start off with introduction trial
introduction_trial.present();
introduction_picture.add_part(intro_part2, 0, 150);
introduction_trial.present();
introduction_picture.add_part(intro_part3, 0, 0);
introduction_trial.present();
introduction_picture.add_part(intro_part4, 0, -150);
introduction_trial.present();

introduction_trial2.present();

# Wait for scanner (until pulse = 2)
pulsetrial.present();
int currentpulse_count = pulse_manager.main_pulse_count();
loop until (pulse_manager.main_pulse_count()-currentpulse_count)>1 begin end;

int timer = clock.time();

# Start default screen (black) and audio file!
default.present();
narrative.present();

# Print onset of narrative to logfile
out.print(string(timer));	out.print("\n");

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
			if (stimulus_category == 10) then exp_pic.add_part(face1_bm, 0, 0);
				elseif (stimulus_category == 11) then exp_pic.add_part(face2_bm, 0, 0);	
				elseif (stimulus_category == 12) then exp_pic.add_part(face3_bm, 0, 0);
				elseif (stimulus_category == 20) then exp_pic.add_part(house1_bm, 0, 0);	
				elseif (stimulus_category == 21) then exp_pic.add_part(house2_bm, 0, 0);
				elseif (stimulus_category == 22) then exp_pic.add_part(house3_bm, 0, 0);	
			end;
			
			# Set to the right location
			exp_pic.set_part_x(it, loc_special[it]);
			exp_event.set_event_code(string(stimulus_category));
			
			# Wait until next presentation
			loop until (clock.time() - timer) > actual_onset begin end;
			
			exp_pic.present();
			
			out.print(current);									out.print("\t");
			out.print(string(stimulus_category+1000));	out.print("\t");
			out.print(string(clock.time()));					out.print("\t");
			out.print(string(clock.time()-timer));			out.print("\n");				
			
			# codes > 1000 are nar-special trials
			tracker.send_message(string(stimulus_category+1000)); 
			
			if (i_nar < 3) then
				actual_onset = nar_onset[i_nar+1];
			else
				int stop_now = clock.time();
				loop until (clock.time() > stop_now + 3000) begin end;
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
		if idx == 0 then idx = 6; end;
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
		actual_onset = actual_onset + 1000;
	elseif current == "nar" then
		stimulus_category = stimulus_category + 100; # to disentangle from iso_stim
	end;
	
	loop until (clock.time() - timer) > actual_onset begin end;
	start_pic = clock.time();
	
	# Run experimental trial until stim_duration has ended
	exp_event.set_event_code(string(stimulus_category));
	exp_trial.present();
		
	out.print(current); 								out.print("\t");
	out.print(string(stimulus_category));		out.print("\t");
	out.print(string(clock.time()));				out.print("\t");
	out.print(string(clock.time()-timer));		out.print("\n");						
	
	tracker.send_message(string(stimulus_category)); 

	loop until clock.time() > (start_pic + stim_dur) begin end;
	
	# Remove stimulus from exp_event
	exp_pic.remove_part(1);
	
	# Got back to default screen
	default.present();
	
	if i_iso == (iso_onset.count()+1) then stop = 1; end; 
	
end;

include "tracker_wrapup.pcl"

end_trial.present();