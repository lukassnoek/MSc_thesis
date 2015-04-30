# --- Dynamic Affect: NARRATIVE --- #

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

response_matching = simple_matching;
active_buttons = 3;
button_codes = 1,2,3;
response_logging = log_all;
no_logfile = false;   

begin; 

# --------------- SDL definitions: objects --------------- #

# Introduction text (example)
text {
	caption = "Goed opletten en niet in slaap vallen! \n
				  Druk op een knop om door te gaan."; 
	font =		"arial";		
	font_size = 20;
} introduction;

# Narrative audio
wavefile { filename = "TomSawyer_large.wav";} narrative_file;

# Experimental bitmap
bitmap {	filename = "face1.jpg";} face1_bm;
bitmap {	filename = "face2.jpg";} face2_bm;
bitmap {	filename = "face3.jpg";} face3_bm;
bitmap {	filename = "Example_house1.jpg";} house1_bm;
bitmap {	filename = "Example_house2.jpg";} house2_bm;
bitmap {	filename = "Example_house3.jpg";} house3_bm;

# End
text {
			caption = 	"Dit is het einde van dit deel van het experiment."; 
			font =		"arial";		
			font_size = 16;
} end;

# --------------- SDL definitions: PICTURES/SOUNDS --------------- #

# Intro
picture {text introduction; x = 0; y = 0;} introduction_picture;

# Narrative
sound { wavefile narrative_file;
		  loop_playback = true;} narrative;

# End
picture {text end; x = 0; y = 0;} end_picture;

picture {text { caption = "Wacht op scanner"; font="arial"; font_size = 20;}; x=0; y=0;} pulsetrial;

# Background
picture {background_color = 0,0,0;} background;

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

# End trial
trial {
	trial_duration = 3000; 
	stimulus_event {
		picture end_picture;
	} end_event;
} end_trial;

trial {
   picture askMe;
} getOutput;

#--------------- BEGIN PCL --------------------#

begin_pcl; 

# Met dank aan Diane Roozendaal
getOutput.present();
string filename = system_keyboard.get_input(askMe,text2);

output_file out = new output_file;
out.open (filename);

array <int> visual_stim[18] =
	{10, 10, 10,
	 11, 11, 11,
	 12, 12, 12,
	 20, 20, 20,
	 21, 21, 21,
	 22, 22, 22};
visual_stim.shuffle();

array <int> onset[18] = 
	{3000, 8000, 13000, 18000, 23000,
	 28000, 33000, 38000, 43000, 48000,
	 53000, 58000, 63000, 68000, 73000,
	 78000, 83000, 88000};

introduction_trial.present();

pulsetrial.present();
logfile.add_event_entry("Pulsetrial");
int currentpulse_count = pulse_manager.main_pulse_count();
loop until (pulse_manager.main_pulse_count()-currentpulse_count)>1 begin end;

int i = 1;
int stop = 0;
int stimulus_category = 0;
int start_pic = 0;

background.present();
narrative.present();
int timer = clock.time();

if i == 1 then
	out.print("Stim_category");
	out.print("\t");
	out.print("Times");
	out.print("\n");
	out.print("Pulse1");
	out.print("\t");
	out.print(string(timer));
	out.print("\n");
end;

loop until stop == 1 begin;
	stimulus_category = visual_stim[i];
		
	if (stimulus_category == 10) then background.add_part(face1_bm, 0, 0);
	elseif (stimulus_category == 11) then background.add_part(face2_bm, 0, 0);	
	elseif (stimulus_category == 12) then background.add_part(face3_bm, 0, 0);
	elseif (stimulus_category == 20) then background.add_part(house1_bm, 0, 0);	
	elseif (stimulus_category == 21) then background.add_part(house2_bm, 0, 0);
	elseif (stimulus_category == 22) then background.add_part(house3_bm, 0, 0);
	end;
	
	loop until (clock.time() - timer) > onset[i] begin end;
	start_pic = clock.time();
	
	out.print(stimulus_category);
	out.print("\t");
	out.print(start_pic);
	out.print("\n");
	
	background.present();
	logfile.add_event_entry(string(stimulus_category));
	
	loop until clock.time() > (start_pic+2000) begin end;
		
	background.remove_part(1);
	default.present();
	i = i + 1;
	
	if i == visual_stim.count() then stop = 1; end; 
	
end;

default.present(); 
start_pic = clock.time();
loop until clock.time() > (start_pic + 3000) begin end;

end_trial.present();
