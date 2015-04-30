# --- Dynamic Affect: Pretest --- #

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
	caption = 	
	"In dit deel van het experiment krijg je een aantal afbeeldingen meerdere keren te zien. \n
	Na sommige afbeeldingen zal je worden gevraagd om aan te geven hoe positief of negatief jij deze
	afbeelding ervaart, van 'zeer negatief' tot 'zeer positief'. \n
	Je kan het keuze-hokje als volgt bewegen met je rechterhand: \n
		Naar LINKS: wijsvinger \n
		Naar RECHTS: ringvinger \n
		BEVESTIGEN: middelvinger \n
		
	Druk op één van de knoppen om met de taak te beginnen."; 
			font =		"arial";		
			font_size = 20;
			
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
			font_size = 16;} 
			end;

# Choice box
line_graphic { 
	coordinates = -100.0, 50.0, 100.0, 50.0;
	coordinates = -100.0, -50.0, 100.0, -50.0;
	coordinates = -100.0, 54.0, -100.0, -54.5;
	coordinates =  100.0, 54.0, 100.0, -54.5;
	line_width = 10;
} choice_box;

# --------------- SDL definitions: PICTURES --------------- #

# ISI
picture {background_color = 0,0,0;} ISI_picture;

# Intro
picture {text introduction; x = 0; y = 0;} introduction_picture;

picture {text { caption = "Wacht op scanner"; font="arial"; font_size = 20;}; x=0; y=0;} pulsetrial;

# End
picture {text end; x = 0; y = 0;} end_picture;

# Experimental picture
picture {bitmap exp_bitmap; x = 0; y = 0;} experimental_picture;
	
# Evaluation text
picture {
	text { caption = "Hoe positief of negatief vond je het vorige plaatje?"; } instructie; x = 0; y = 200;
   text { caption = "Heel negatief"; } heelneg; x = -500; y = 0;
   text { caption = "Negatief"; } beetjeneg; x = -250; y = 0;
   text { caption = "Neutraal"; } neu; x = 0; y = 0;
   text { caption = "Positief"; } beetjepos; x = 250; y = 0;
   text { caption = "Heel positief"; } heelpos; x = 500; y = 0;
} evaluation_text;

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
	trial_duration = 1000000; 
	stimulus_event {
		picture end_picture;
		delta_time = 2000;
	} end_event;
} end_trial;

# --------------- START OF PCL --------------- #
begin_pcl;

# Regular execution of intro trial
introduction_trial.present();

pulsetrial.present();
int currentpulse_count = pulse_manager.main_pulse_count();
loop until pulse_manager.main_pulse_count()-currentpulse_count>1 begin                   #waits for a pulse and shows instructions
end;

# Arrays with stimuli
array <string> face_stim[3] =
	{"face1.jpg", 
	 "face2.jpg", 
    "face3.jpg"};

array <string> house_stim[3] =
	{"Example_house1.jpg", 
	 "Example_house2.jpg", 
	 "Example_house3.jpg"};

# This array (randomly) lists conditions which correspond to the stimuli (no NULL)
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
array <int> face1_eval[8] = {1, 1, 1, 1, 0, 0, 0, 0}; face1_eval.shuffle();
array <int> face2_eval[8] = {1, 1, 1, 1, 0, 0, 0, 0}; face2_eval.shuffle();
array <int> face3_eval[8] = {1, 1, 1, 1, 0, 0, 0, 0}; face3_eval.shuffle();
array <int> house1_eval[8] = {1, 1, 1, 1, 0, 0, 0, 0}; house1_eval.shuffle();
array <int> house2_eval[8] = {1, 1, 1, 1, 0, 0, 0, 0}; house2_eval.shuffle();
array <int> house3_eval[8] = {1, 1, 1, 1, 0, 0, 0, 0}; house3_eval.shuffle();

# Initial position of box (x - coordinate)
array <int> box_position[5] =
	{-500, -250, 0, 250, 500};

int stimulus_category = 0;
int i_f1 = 1; int i_f2 = 1; int i_f3 = 1;
int i_h1 = 1; int i_h2 = 1; int i_h3 = 1;
int go_eval = 0;
int ISI = 6000; # Separately defined here
int start_pic = 0;

# --- start the loop of the experimental_trial --- #	
loop int i = 1 until i > stimuli_condition.count() begin;

	stimulus_category = stimuli_condition[i]; # random condition is drawn
	box_position.shuffle(); # Random starting position is draw (as first index, later)
	
	# Condition = face1
	if (stimulus_category == 10) then
		start_pic = clock.time();
		
		exp_bitmap.unload();
		exp_bitmap.set_filename(face_stim[1]);
		exp_bitmap.load();
		
		experimental_event.set_event_code(string(stimulus_category));
		
		experimental_trial.present();
		ISI_picture.present();
		logfile.add_event_entry("ISI");
		loop until clock.time() > (start_pic+2000+ISI) begin end;
		
		if face1_eval[i_f1] == 1 then go_eval = 1; else go_eval = 0; end;
		i_f1 = i_f1 + 1;
		
	# Condition = face2
	elseif (stimulus_category == 11) then
		start_pic = clock.time();
		
		exp_bitmap.unload();
		exp_bitmap.set_filename(face_stim[2]);
		exp_bitmap.load();
		
		experimental_event.set_event_code(string(stimulus_category));
		experimental_trial.present();
		ISI_picture.present();
		logfile.add_event_entry("ISI");
		loop until clock.time() > (start_pic+2000+ISI) begin end;
		
		if face2_eval[i_f2] == 1 then go_eval = 1; else go_eval = 0; end;
		i_f2 = i_f2 + 1;
		
	# Condition = face3
	elseif (stimulus_category == 12) then
		start_pic = clock.time();
		
		exp_bitmap.unload();
		exp_bitmap.set_filename(face_stim[3]);
		exp_bitmap.load();
		
		experimental_event.set_event_code(string(stimulus_category));
		experimental_trial.present();
		ISI_picture.present();
		logfile.add_event_entry("ISI");
		loop until clock.time() > (start_pic+2000+ISI) begin end;

		if face3_eval[i_f3] == 1 then go_eval = 1; else go_eval = 0; end;
		i_f3 = i_f3 + 1;
		
	# Condition = house1
	elseif (stimulus_category == 20) then
		start_pic = clock.time();
		
		exp_bitmap.unload();
		exp_bitmap.set_filename(house_stim[1]);
		exp_bitmap.load();
		
		experimental_event.set_event_code(string(stimulus_category));
		experimental_trial.present();
		ISI_picture.present();
		logfile.add_event_entry("ISI");
		loop until clock.time() > (start_pic+2000+ISI) begin end;
	
		if house1_eval[i_h1] == 1 then go_eval = 1; else go_eval = 0; end;
		i_h1 = i_h1 + 1;
		
	# Condition = house2
	elseif (stimulus_category == 21) then
		start_pic = clock.time();
		
		exp_bitmap.unload();
		exp_bitmap.set_filename(house_stim[2]);
		exp_bitmap.load();
		
		experimental_event.set_event_code(string(stimulus_category));
		experimental_trial.present();
		ISI_picture.present();
		logfile.add_event_entry("ISI");
		loop until clock.time() > (start_pic+2000+ISI) begin end;
	
		if house2_eval[i_h2] == 1 then go_eval = 1; else go_eval = 0; end;
		i_h2 = i_h2 + 1;
		
	# Condition = house3
	elseif (stimulus_category == 22) then
		start_pic = clock.time();
		
		exp_bitmap.unload();
		exp_bitmap.set_filename(house_stim[3]);
		exp_bitmap.load();
		
		experimental_event.set_event_code(string(stimulus_category));
		experimental_trial.present();
		ISI_picture.present();
		logfile.add_event_entry("ISI");
		loop until clock.time() > (start_pic+2000+ISI) begin end;
		
		if house3_eval[i_h3] == 1 then go_eval = 1; else go_eval = 0; end;
		i_h3 = i_h3 + 1;
		
	# Condition = NULL-event
	#elseif (sentence_category == 4) then
	#		start_null = clock.time();
	#		ITI_picture.present();
	#		logfile.add_event_entry("NULL");
	#		loop until clock.time() > (start_null+10000) begin end;
					
	end; # end of if-statement
	
	int confirm = 0;
	int int_pos = 0;
	int toMove = 0;
	int j = 1; 
		
	if go_eval == 1 then
			
		loop until confirm == 1 begin
		
			if j == 1 then
				int_pos = box_position[1];
				evaluation_text.add_part(choice_box,int_pos,0);
				evaluation_trial.present();
			end;
		
			if response_manager.last_response() == 1  then
				toMove = int_pos - 250;
			
				if toMove < - 500 then toMove = -500; end;
	
				evaluation_text.remove_part(7);
				evaluation_text.add_part(choice_box,toMove,0);
				evaluation_trial.present();
		
			elseif response_manager.last_response() == 3 then
				toMove = int_pos + 250;
		
				if toMove > 500 then toMove = 500; end;
	
				evaluation_text.remove_part(7);
				evaluation_text.add_part(choice_box,toMove,0);
				evaluation_trial.present();
		
			elseif response_manager.last_response() == 2 then
				start_pic = clock.time();
				choice_box.set_line_color(0, 200, 0, 255);
				choice_box.redraw();
				confirm = 1;
				
				stop_event.set_event_code(string(toMove));
				stop_trial.present();
				evaluation_text.remove_part(7);
				ISI_picture.present();
				logfile.add_event_entry("ISI");
				loop until clock.time() > (start_pic+2000) begin end;
			end;

			int_pos = toMove;
			j = j + 1; 
			
		end;

	end;

	i = i + 1;

choice_box.set_line_color(255, 255, 255, 255);	
choice_box.redraw();

end; 

end_trial.present();


