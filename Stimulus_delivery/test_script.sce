
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

# Choice box
line_graphic  {
#			coordinates = -100, 50, 100, 50;
#			coordinates = -100, -50, 100, -50;
#			coordinates = -100, 54.5, -100, -54.5;
#			coordinates = 100, 54.5, 100, -54.5;
#			line_width = 10;
} 
choice_box;

# Text
picture {
   text { caption = "Heel negatief"; };
   x = -500; y = 0;
   text { caption = "Beetje negatief"; };
   x = -250; y = 0;
   text { caption = "Neutraal"; };
   x = 0; y = 0;
   text { caption = "Beetje positief"; };
   x = 250; y = 0;
   text { caption = "Heel positief"; };
   x = 500; y = 0;
} evaluation_text;

picture {
	line_graphic choice_box; x = 0; y = 0;} 
choice_pic;

trial {
	trial_duration = forever;
	trial_type = specific_response;  
   terminator_button = 3;
	stimulus_event {
		picture evaluation_text;
	} text_event;
} text_trial;

begin_pcl;

array <int> box_position[5] =
	{-500, -250, 0, 250, 500};

box_position.shuffle();

array <double> box_coord[16] = 
	{-100, 50, 100, 50,
	 -100, -50, 100, -50,
	 -100, 54.5, -100, -54.5,
	 100, 54.5, 100, -54.5};

choice_box.add_line(box_coord[1], box_coord[2], box_coord[3], box_coord[4]);
choice_box.add_line(box_coord[5], box_coord[6], box_coord[7], box_coord[8]);
choice_box.add_line(box_coord[9], box_coord[10], box_coord[11], box_coord[12]);
choice_box.add_line(box_coord[13], box_coord[14], box_coord[15], box_coord[16]);

choice_box.set_line_color( 255, 255, 255, 255 );
choice_box.set_line_width(10);
choice_box.redraw();

evaluation_text.add_part(choice_box,box_position[1],0);
text_trial.present();

if (response_manager.last_response() == 2) then
	choice_box.set_line_color( 0, 255, 0, 255 );
	choice_box.redraw();
	evaluation_text.add_part(choice_box,box_position[1],0);
end;

text_trial.present();

