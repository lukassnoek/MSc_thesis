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
	"Tijdens deze taak krijg je een aantal afbeeldingen te zien.
	Kijk hier goed naar en houd je aandacht erbij.
	Blijf tussen de afbeeldingen in naar het fixatie-kruis (+) kijken. \n \n
	Druk op een knop om te beginnen.";
} introduction;

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

# --------------- SDL definitions: PICTURES --------------- #

# ISI
picture {text {caption = "+"; font_size = 40; background_color = 0,0,0;}; x=0; y=0;} ISI_picture;

# Intro
picture {text introduction; x = 0; y = 0;} introduction_picture;

picture {text { caption = "We gaan bijna beginnen! \n Even wachten op de scanner..."; font="arial"; font_size = 20;}; x=0; y=0;} pulsetrial;

# End
picture {text end; x = 0; y = 0;} end_picture;

# Experimental picture
picture {bitmap exp_bitmap; x = 0; y = 0;} experimental_picture;
	
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

# Experimental trial
trial {
	stimulus_event {
		picture experimental_picture;
		duration = 2000;
	} experimental_event;
} experimental_trial; 

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
string filename_suffix = "s";
include "DA_tracker_init.pcl"

# --------------- (STIMULI) ARRAYS ----------- #

# Stimuli
array <string> exp_stimuli[3] =
	{"char_janos.jpg", 
	 "char_matthias.jpg", 
    "char_benji.jpg"};

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
out.open (filename + "_stimulusdriven.txt");

out.print("Cat"); 	out.print("\t");	
out.print("T_abs"); 	out.print("\t");
out.print("T_rel"); 	out.print("\t");
	
int confirm = 0;
int int_pos = 0;
int toMove = 1000;
int j = 1; 

introduction_trial.present();

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

include "tracker_wrapup.pcl"

end_trial.present();