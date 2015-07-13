# --- Dynamic Affect: Posttest --- #
# Experimental script to run the Posttest scenario
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

#scenario_type = fMRI;
scenario_type = fMRI_emulation;
scan_period = 2000;
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

# Introduction text (example)

text {
	caption =
	"Tijdens de taak is het belangrijk dat je naar het fixatie-kruis (+) blijft kijken
	tussen de presentaties van de afbeeldingen. Ook is het belangrijk dat je je ogen
	open houdt en niet in slaap valt! \n \n
	Druk op een knop om te beginnen.";
} introduction3;

# Experimental bitmap
bitmap {
			filename = "";
			preload = false;
} exp_bitmap;

# End
text {
			caption = 	"Dit is het einde van dit deel van het experiment."; 
			font =		"arial";		
			font_size = 25;} 
			end;

# Choice box
line_graphic { 
	coordinates = -120.0, 50.0, 120.0, 50.0;
	coordinates = -120.0, -50.0, 120.0, -50.0;
	coordinates = -120.0, 54.0, -120.0, -54.5;
	coordinates =  120.0, 54.0, 120.0, -54.5;
	line_width = 10;
} choice_box;

# --------------- SDL definitions: PICTURES --------------- #

# ISI
picture {text {caption = "+"; font_size = 40; background_color = 0,0,0;}; x=0; y=0;} ISI_picture;

# Intro
picture {text introduction3; x = 0; y = 0;} introduction_picture3;

picture {text { caption = "We gaan bijna beginnen! \n Even wachten op de scanner..."; font="arial"; font_size = 20;}; x=0; y=0;} pulsetrial;

# End
picture {text end; x = 0; y = 0;} end_picture;

# Experimental picture
picture {bitmap exp_bitmap; x = 0; y = 0;} experimental_picture;
	
# Evaluation text
picture {
	text { caption = "Hoe positief of negatief vond je het vorige plaatje?"; font_size = 40;} instructie; x = 0; y = 200; 
	text { caption = "Heel erg negatief"; font_size = 20;} heelergneg; x = -750; y = 0;
   text { caption = "Heel negatief"; font_size = 20;} heelneg; x = -500; y = 0;
   text { caption = "Enigszins negatief"; font_size = 20;} enigneg; x = -250; y = 0;
   text { caption = "Neutraal"; font_size = 20;} neu; x = 0; y = 0;
   text { caption = "Enigszins positief"; font_size = 20;} enigpos; x = 250; y = 0;
	text { caption = "Heel positief"; font_size = 20;} heelpos; x = 500; y = 0;
   text { caption = "Heel erg positief"; font_size = 20;} heelergpos; x = 750; y = 0;
} evaluation_text;

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

# Log-trial
trial {
   picture askMe;
} getOutput;

# Intro trial3
trial {
	all_responses		= true; 
	trial_type 			= first_response;
	trial_duration 	= forever;
		stimulus_event {
		picture introduction_picture3;
		time = 0;
		} introduction_event3;
} introduction_trial3;

# Experimental trial
trial {
	stimulus_event {
		picture experimental_picture;
		duration = 2000;
	} experimental_event;
} experimental_trial;  

# Evaluation trial: GO
trial {
	trial_duration = forever;
	trial_type = specific_response;
	terminator_button = 1,2,3;
	
	stimulus_event {
		picture evaluation_text; 
	} text_event;
} evaluation_trial;

# Evaluation trial: STOP
trial {
	trial_duration = 500;
	
	stimulus_event {
		picture evaluation_text; 
	} stop_event;
} stop_trial;

# End trial
trial {
	trial_duration = 5000; 
	stimulus_event {
		picture end_picture;
		delta_time = 2000;
	} end_event;
} end_trial;

# --------------- START OF PCL --------------- #
begin_pcl;

# --------------- EYETRACKING STUFF ---------- #
# Some parameter for the eyetracker.
#

# just testing? (1 = yes)
int trk_test = 1; 

# Suffix for .edf filename
string filename_suffix = "p";
include "DA_tracker_init.pcl"

# --------------- (STIMULI) ARRAYS ----------- #

# Stimuli
array <string> face_stim[3] =
	{"char_janos.jpg", 
	 "char_matthias.jpg", 
    "char_benji.jpg"};

array <string> house_stim[3] =
	{"loc_janos.jpg", 
	 "loc_matthias.jpg", 
	 "loc_benji.jpg"};

# Array (randomly) listing conditions which correspond to the stimuli (no NULL)
array <int> stimuli_condition[48] =
	{	10, 10, 10, 10, 10, 10, 10, 10,
		11, 11, 11, 11, 11, 11, 11, 11,
		12, 12, 12, 12, 12, 12, 12, 12,
		20, 20, 20, 20, 20, 20, 20, 20,
		21, 21, 21, 21, 21, 21, 21, 21, 
		22, 22, 22, 22, 22, 22, 22, 22
	}; # 10-12 = face, 20-22 = house

stimuli_condition.shuffle(); # randomization

# Array for random selection of evaluation for each trial
# 0 = no evaluation, 1 = evaluation
array <int> face1_eval[8] = {1, 1, 1, 1, 0, 0, 0, 0}; face1_eval.shuffle();
array <int> face2_eval[8] = {1, 1, 1, 1, 0, 0, 0, 0}; face2_eval.shuffle();
array <int> face3_eval[8] = {1, 1, 1, 1, 0, 0, 0, 0}; face3_eval.shuffle();
array <int> house1_eval[8] = {1, 1, 1, 1, 0, 0, 0, 0}; house1_eval.shuffle();
array <int> house2_eval[8] = {1, 1, 1, 1, 0, 0, 0, 0}; house2_eval.shuffle();
array <int> house3_eval[8] = {1, 1, 1, 1, 0, 0, 0, 0}; house3_eval.shuffle();

# Initial position of choice-box (x - coordinate)
array <int> box_position[7] =
	{-750, -500, -250, 0, 250, 500, 750};

# Some variable definitions/initializations
int stimulus_category = 0; 

int i_f1 = 1; int i_f2 = 1; int i_f3 = 1; # face-counters
int i_h1 = 1; int i_h2 = 1; int i_h3 = 1; # hosue-coutners

# Value for evaluation yes/no (0 = no)
int go_eval = 0;
int ISI = 6000; # Separately defined here
int start_pic = 0;
int onsett = 0;
int timer = 0;

# Ask for logfile-name
getOutput.present();
string filename = system_keyboard.get_input(askMe,text2);

output_file out = new output_file;
out.open (filename + "_post.txt");

out.print("Cat"); 	out.print("\t");	
out.print("T_abs"); 	out.print("\t");
out.print("T_rel"); 	out.print("\t");
out.print("Val"); 	out.print("\n");	
	
int confirm = 0;
int int_pos = 0;
int toMove = 1000;
int j = 1; 

introduction_trial3.present();

# --------------- WAIT FOR PULSE-trial ------------- #
pulsetrial.present();
int currentpulse_count = pulse_manager.main_pulse_count();
loop until pulse_manager.main_pulse_count()-currentpulse_count > 1 begin end;

# timer = start of experiment
timer = clock.time();
tracker.send_message("start_exp");

# Write starting time
out.print("Start_exp");		out.print("\t");
out.print(string(timer)); 	out.print("\t");
out.print(string(0));	 	out.print("\t");
out.print(string(0)); 		out.print("\n");

# Wait 5 secs
ISI_picture.present();
loop until clock.time() > (timer + 5000) begin end;

# --------------- MAIN EXPERIMENTAL LOOP ------------- #
# Loops until all stimuli have been presented. 
# In 50% of the trials (counterbalenced over conditions),
# an evaluation-trial is presented.

loop int i = 1 until i > stimuli_condition.count() begin;

	stimulus_category = stimuli_condition[i]; # random condition is drawn
	box_position.shuffle(); # Random starting position is draw (as first index, later)
	
	exp_bitmap.unload();
	# Condition = face1
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

	exp_bitmap.load();
	start_pic = clock.time();
	experimental_event.set_event_code(string(stimulus_category));
	experimental_trial.present();
	onsett = clock.time() - 2000; # onset minus duration
		
	out.print(string(stimulus_category)); out.print("\t");
	out.print(string(onsett)); out.print("\t");
	out.print(string(onsett-timer)); out.print("\t");
	tracker.send_message(string(stimulus_category));
 
	ISI_picture.present();
	loop until clock.time() > (start_pic+2000+ISI) begin end;
		
	if go_eval == 0 then out.print("\n"); end;
	
	# Resetting variables from practice trials
	confirm = 0;
	int_pos = 0;
	toMove = 1000;
	j = 1; 
		
	if go_eval == 1 then
			
		loop until confirm == 1 begin
		
			if j == 1 then
				int_pos = box_position[1];
				evaluation_text.add_part(choice_box,int_pos,0);
				evaluation_trial.present();
			end;
		
			if response_manager.last_response() == 1  then
				toMove = int_pos - 250;
			
				if toMove < - 750 then toMove = -750; end;
	
				evaluation_text.remove_part(9);
				evaluation_text.add_part(choice_box,toMove,0);
				evaluation_trial.present();
		
			elseif response_manager.last_response() == 3 then
				toMove = int_pos + 250;
		
				if toMove > 750 then toMove = 750; end;
	
				evaluation_text.remove_part(9);
				evaluation_text.add_part(choice_box,toMove,0);
				evaluation_trial.present();
		
			elseif response_manager.last_response() == 2 then
				start_pic = clock.time();
				
				# Add 1000 to position and divide by interval (250) to transform
				# the position of box into likert-scale from 1 - 7
				if toMove == 1000 then
					out.print(string((int_pos+1000)/250));
					stop_event.set_event_code(string((int_pos+1000)/250));
				else
					out.print(string((toMove+1000)/250));
					stop_event.set_event_code(string((toMove+1000)/250));
				end;
				
				out.print("\n");
				
				choice_box.set_line_color(0, 200, 0, 255);
				choice_box.redraw();
				confirm = 1;
				
				stop_trial.present();
				evaluation_text.remove_part(9);
				ISI_picture.present();
				logfile.add_event_entry("ISI");
				loop until clock.time() > (start_pic+2000) begin end;
			end;

			int_pos = toMove;
			j = j + 1; 
			
		end;

	end;

	i = i + 1;

	# Reset line color of box to white
	choice_box.set_line_color(255, 255, 255, 255);	
	choice_box.redraw();

end; 

include "tracker_wrapup.pcl"

end_trial.present();