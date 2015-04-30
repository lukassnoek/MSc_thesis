

scenario="Face_Moving_MD";

no_logfile = false;
#write_codes = true;
#pulse_width = 20;
#response_port_output = false;

response_matching = simple_matching;

screen_width = 1366;
screen_height = 768;
screen_bit_depth = 24;

active_buttons = 4;    
button_codes = 1,2,3,4;

default_font_size=26;
default_font = "Arial";
default_text_color = 255,255,255;
default_background_color = 150,150,150;

begin;

# define the placeholder bitmaps
bitmap { filename="D45_1.bmp"; } D_placeholder;
bitmap { filename="face40_1.bmp"; } F_placeholder; # huis of gezicht


# array met de achtergronden
array {
   LOOP $i 4;

		$k = '$i + 1';

			bitmap { filename = "D45_$k.bmp";};
			
		
   ENDLOOP;
} D45;


# arrays met de frames
#faces
	array{
			bitmap { filename = "face40_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face40_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face40_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face40_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face41_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face41_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face41_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face41_4.bmp"; trans_src_color = 86, 91, 119; };			
			bitmap { filename = "face42_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face42_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face42_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face42_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face43_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face43_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face43_3.bmp"; trans_src_color = 86, 91, 119; };			
			bitmap { filename = "face43_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face44_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face44_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face44_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face44_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face45_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face45_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face45_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face45_4.bmp"; trans_src_color = 86, 91, 119; };			
			bitmap { filename = "face46_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face46_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face46_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face46_4.bmp"; trans_src_color = 86, 91, 119; };			
			bitmap { filename = "face47_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face47_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face47_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face47_4.bmp"; trans_src_color = 86, 91, 119; };			
			bitmap { filename = "face48_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face48_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face48_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face48_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face49_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face49_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face49_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face49_4.bmp"; trans_src_color = 86, 91, 119; };


		}F1_x;
		
	array{
			bitmap { filename = "face8_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face8_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face8_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face8_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face9_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face9_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face9_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face9_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face20_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face20_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face20_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face20_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face21_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face21_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face21_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face21_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face24_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face24_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face24_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face24_4.bmp"; trans_src_color = 86, 91, 119; };#
			bitmap { filename = "face25_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face25_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face25_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face25_4.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face36_1.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face36_2.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face36_3.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face36_4.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face37_1.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face37_2.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face37_3.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face37_4.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face38_1.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face38_2.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face38_3.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face38_4.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face39_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face39_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face39_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face39_4.bmp"; trans_src_color = 86, 91, 119; };

		}F2_x;
		
	array{
			bitmap { filename = "face23_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face23_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face23_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face23_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face26_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face26_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face26_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face26_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face28_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face28_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face28_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face28_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face29_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face29_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face29_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face29_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face30_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face30_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face30_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "face30_4.bmp"; trans_src_color = 86, 91, 119; };#
			bitmap { filename = "face31_1.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face31_2.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face31_3.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face31_4.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face32_1.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face32_2.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face32_3.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face32_4.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face33_1.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face33_2.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face33_3.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face33_4.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face34_1.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face34_2.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face34_3.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face34_4.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face35_1.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face35_2.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face35_3.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "face35_4.bmp"; trans_src_color = 87, 91, 119; };

		}F3_x;
		





#Non face

array{
			bitmap { filename = "niks24_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks24_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks24_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks24_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks25_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks25_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks25_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks25_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks26_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks26_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks26_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks26_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks27_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks27_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks27_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks27_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks28_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks28_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks28_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks28_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks29_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks29_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks29_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks29_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks30_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks30_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks30_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks30_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks31_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks31_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks31_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks31_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks32_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks32_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks32_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks32_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks33_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks33_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks33_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks33_4.bmp"; trans_src_color = 86, 91, 119; };
} H1_x;

array{
			bitmap { filename = "niks4_1.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "niks4_2.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "niks4_3.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "niks4_4.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "niks6_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks6_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks6_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks6_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks10_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks10_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks10_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks10_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks11_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks11_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks11_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks11_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks14_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks14_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks14_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks14_4.bmp"; trans_src_color = 86, 91, 119; };			
			bitmap { filename = "niks19_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks19_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks19_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks19_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks20_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks20_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks20_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks20_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks21_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks21_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks21_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks21_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks22_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks22_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks22_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks22_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "house22_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "house22_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "house22_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "house22_4.bmp"; trans_src_color = 86, 91, 119; };
} H2_x;

array{
			bitmap { filename = "house2_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "house2_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "house2_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "house2_4.bmp"; trans_src_color = 86, 91, 119; };			
			bitmap { filename = "house20_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "house20_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "house20_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "house20_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks5_1.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "niks5_2.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "niks5_3.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "niks5_4.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "niks9_1.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "niks9_2.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "niks9_3.bmp"; trans_src_color = 87, 91, 119; };			
			bitmap { filename = "niks9_4.bmp"; trans_src_color = 87, 91, 119; };
			bitmap { filename = "niks13_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks13_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks13_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks13_4.bmp"; trans_src_color = 86, 91, 119; };			
			bitmap { filename = "niks15_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks15_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks15_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks15_4.bmp"; trans_src_color = 86, 91, 119; };	
			bitmap { filename = "niks16_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks16_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks16_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks16_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks17_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks17_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks17_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks17_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks18_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks18_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks18_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks18_4.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks23_1.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks23_2.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks23_3.bmp"; trans_src_color = 86, 91, 119; };
			bitmap { filename = "niks23_4.bmp"; trans_src_color = 86, 91, 119; };
} H3_x;



# 6 arrays : moilijk/makkelijk/middel voor F/H >60



picture {
	background_color=128,128,128;
	
	box {
		color=255,0,0;
		height=6;
		width=6; };
		x=0;
		y=0;
} default;

# define the pictures

# fixation dot: just a small red dot in the middle of the screen

picture {
	#background_color=0,0,0;
	
	box {
		color=255,0,0;
		height=6;
		width=6; };
		x=0;
		y=0;
} fix;

# a picture for text input (getFilename)
picture {
	text { 
		caption = "Enter the output filename:"; font_color=255,255,255; };
		x = 0; y = 100;
		text { 
			caption = " "; } text2;
		x = 0; y = 0;
} askMe;

picture {
	text {
		caption = "Welcome to the experiment: 
					'Do I see a face'?
					In this experiment you will see a face or something else
					You need to respond with your left index finger (x) at faces
					and with your right index finger (.) if you didn't see a face

					Press a button to continue";}; x=0;y=0;
	}welkomtext;
picture {
	text { 
		caption = "After each stimuli you will be asked how certain you were of your choice
							you should use the entire scale of 1 till 4 
										
		1 (z) = very uncertain         2 (x) = uncertain         3 (.) = certain         4 (/) = very certain

             Press a button to continue";}; x=0; y=0; 
	 }zekertext;
picture{
	text { 
		caption = "You have 5 seconds to respond to each question.
				Half of the stimuli are faces, the other half are non-faces.
					Press a button to start the experiment.
							Good luck!";}; x=0;y=0;
	}succestext;



picture { 	
	text { 
		caption = "Was it a face?";}; x = 0; y = 150;
	text{ 
		caption = "yes"; }; x = -174; y = 0;		 
	text{ 
		caption = "no"; }; x = 174; y = 0; 
		
}huisgezichtVraag;

picture { 	
	text { 
		caption = "Was it a face?";}; x = 0; y = 150;
	text{ 
		caption = "yes"; font_color=0,0,255;}; x = -174; y = 0;		 
	text{ 
		caption = "no"; font_color=255,255,255;}; x = 174; y = 0; 
		
}gezichtresponse;

picture { 	
	text { 
		caption = "Was it a face?";}; x = 0; y = 150;
	text{ 
		caption = "yes"; font_color=255,255,255;}; x = -174; y = 0;		 
	text{ 
		caption = "no"; font_color=0,0,255;}; x = 174; y = 0; 
		
}huisresponse;

picture {
	text{ 
		caption = "How certain were you of your choice?
				1 = very uncertain   4 = very certain"; font_color=255,255,255; }; x = 0; y = 150;
	text{ 
		caption = "1"; font_color=255,255,255; } zekertext1; x = -378; y = 0;		 
	text{ 
		caption = "2"; font_color=255,255,255; } zekertext2; x = -174; y = 0; 
	text{ 
		caption = "3"; font_color=255,255,255; } zekertext3; x = 174; y = 0;  
	text{ 
		caption = "4"; font_color=255,255,255; } zekertext4; x = 378; y = 0; 
		
	} zekerVraag;

picture {
	text{ 
		caption = "How certain were you of your choice?
				1 = very uncertain   4 = very certain"; font_color=255,255,255; }; x = 0; y = 150;
	text{ 
		caption = "1"; font_color=0,0,255; }; x = -378; y = 0;		 
	text{ 
		caption = "2"; font_color=255,255,255; }; x = -174; y = 0; 
	text{ 
		caption = "3"; font_color=255,255,255; }; x = 174; y = 0;  
	text{ 
		caption = "4"; font_color=255,255,255; }; x = 378; y = 0; 
		
	} antwoord1;

picture {
	text{ 
		caption = "How certain were you of your choice?
				1 = very uncertain   4 = very certain"; font_color=255,255,255; }; x = 0; y = 150;
	text{ 
		caption = "1"; font_color=255,255,255; }; x = -378; y = 0;		 
	text{ 
		caption = "2"; font_color=0,0,255; }; x = -174; y = 0; 
	text{ 
		caption = "3"; font_color=255,255,255; } ; x = 174; y = 0;  
	text{ 
		caption = "4"; font_color=255,255,255; }; x = 378; y = 0; 
		
	} antwoord2;

picture {
	text{ 
		caption ="How certain were you of your choice?
				1 = very uncertain   4 = very certain"; font_color=255,255,255; }; x = 0; y = 150;
	text{ 
		caption = "1"; font_color=255,255,255; }; x = -378; y = 0;		 
	text{ 
		caption = "2"; font_color=255,255,255; }; x = -174; y = 0; 
	text{ 
		caption = "3"; font_color=0,0,255; }; x = 174; y = 0;  
	text{ 
		caption = "4"; font_color=255,255,255; }; x = 378; y = 0; 
		
	} antwoord3;

picture {
	text{ 
		caption = "How certain were you of your choice?
				1 = very uncertain   4 = very certain"; font_color=255,255,255; }; x = 0; y = 150;
	text{ 
		caption = "1"; font_color=255,255,255; }; x = -378; y = 0;		 
	text{ 
		caption = "2"; font_color=255,255,255; }; x = -174; y = 0; 
	text{ 
		caption = "3"; font_color=255,255,255; }; x = 174; y = 0;  
	text{ 
		caption = "4"; font_color=0,0,255; }; x = 378; y = 0; 
		
	} antwoord4;

picture {
	text{ 
		caption = "This is the end of the first part, ask for the experimenter!";}; x=0;y=200;
} accuracy; 		
# dit worden de schermen met de feitelijke stimuli
# NB elke frame is een picture

# fixatiescherm: random dots met groene fix
picture {

	#background_color=0,0,0;
	
	# random dots:
	bitmap D_placeholder;
	x=0;
	y=0;
	
	bitmap F_placeholder;
	x=0;
	y=0;

	
	# de groene fixatie:	
	box {
		color=0,255,0;
		height=6;
		width=6; };
		x=0;
		y=0;

} fixScreenGreen;

# rode fixatie
picture {

	#background_color=0,0,0;
	
	# random dots:
	bitmap D_placeholder;
	x=0;
	y=0;
	
	bitmap F_placeholder;
	x=0;
	y=0;

	
	# de rode fixatie:	
	box {
		color=255,0,0;
		height=6;
		width=6; };
		x=0;
		y=0;

} fixScreenRed;

# motion frame 1
picture {

	#background_color=0,0,0;
	
	# random dots:
	bitmap D_placeholder;
	x=0;
	y=0;
	
	bitmap F_placeholder;
	x=0;
	y=0;

	
	# de rode fixatie:	
	box {
		color=0,255,0;
		height=6;
		width=6; };
		x=0;
		y=0;

} motionFrame1;

# motion frame 2
picture {

	#background_color=0,0,0;
	
	# random dots:
	bitmap D_placeholder;
	x=0;
	y=0;
	
	bitmap F_placeholder;
	x=0;
	y=0;


	
	# de rode fixatie:	
	box {
		color=0,255,0;
		height=6;
		width=6; };
		x=0;
		y=0;

} motionFrame2;

# motion frame 2
picture {

	#background_color=0,0,0;
	
	# random dots:
	bitmap D_placeholder;
	x=0;
	y=0;
	
	bitmap F_placeholder;
	x=0;
	y=0;

	
	# de rode fixatie:	
	box {
		color=0,255,0;
		height=6;
		width=6; };
		x=0;
		y=0;

} motionFrame3;






# description of a standard trial

trial {
	start_delay= 2500;  #2500
	trial_duration=stimuli_length;
	
	stimulus_event {
	picture fixScreenRed;
	time=0;
	} fixRedEvent;
	
	stimulus_event {
	picture fixScreenGreen;
	time= 1750; #1750
	} fixGreenEvent;
	
	stimulus_event {
	picture motionFrame1;
	deltat=250;
	#port_code = 2;
	} motion1Event;

	stimulus_event {
	picture motionFrame2;
	deltat=10;
	} motion2Event;	
	
   stimulus_event {
	picture motionFrame3;
	deltat=10;
	code = "motionEvent3"; 
	} motionEvent3;

   stimulus_event {
	picture motionFrame2;
	deltat=10;
	code = "motionEvent3"; 
	} motionEvent4;
	
	 stimulus_event {
	picture motionFrame1;
	deltat=10;
	code = "motionEvent3"; 
	} motionEvent5;
	
	 stimulus_event {
	picture motionFrame1;
	deltat=10;
	duration=1250;
	code = "motionEvent3"; 
	} still;
	
	#stimulus_event {
	#picture default;
	#time = 1750;
	#} defaultEnd;
	
	
	
	
	#stimulus_event {
	#picture motionFrame3;
	#port_code=255;
	#code = "pulse_event";
	#deltat=50; 						
	#} motionEv3;

#stimulus_event {
#   nothing {};
#  port_code=255;
#	code = "pulse_event";
#   deltat= 50;	
#} pulseEvent;
		# NB -> tijden moeten gecorrigeerd worden voor offset!

	
} mainTrial; 

# text input trial; this has no use for the experiment,
# it is only needed to get the filename you want to save
# the results in.

trial {
   picture askMe;
} getOutput;

trial {
trial_duration = stimuli_length;
	stimulus_event {
		picture welkomtext;
		duration = response;
		code = "welkomtext";};		
	stimulus_event {
		picture zekertext;
		duration = response;
		code = "zekertext";};
	stimulus_event {
		picture succestext; 
		duration = response;
		code = "succes";};
}instructieTrial;

trial {
trial_duration = 5000;
trial_type = specific_response;
terminator_button = 2,3;
	stimulus_event {
		picture huisgezichtVraag;
		response_active = true;
		code = " ";
	} huisgezichtEvent;
} huisgezichtTrial;

trial{
trial_duration = 500;	
		stimulus_event {
		picture  antwoord1;
		response_active=false;
		code ="feedback";
	} feedbackstimEvent;
} feedbackstimTrial;



trial { 
trial_duration = 5000;	
trial_type= first_response;
	stimulus_event {
		picture zekerVraag;
		response_active=true;
		code = " ";
	} zekerEvent;
} zekerTrial;

trial{
trial_duration = 500;	
		stimulus_event {
		picture  antwoord1;
		response_active=false;
		code ="feedback";
	} feedbackzekerEvent;
} feedbackzekerTrial;

trial{
trial_type = first_response;
trial_duration = forever;
	stimulus_event {
		picture accuracy;
		response_active=false;
		code="end";
		} endEvent;
	} endoftask;

# get started picture





# The real experiment starts here:

begin_pcl;


# open output file
getOutput.present();
string filename = system_keyboard.get_input(askMe,text2);

output_file out = new output_file;
out.open (filename);

instructieTrial.present();



	
############# MAIN EXPERIMENT ###############

# array met codes voor de stimuli
# stimulus code codeert voor frame/stack EN voor pulstijd

# codes zijn 1 - 28

# 1-14 -> stacks
# 15-28 -> frames

# pulstijd 20+(n*20) ms voor de stacks
# 20+((n-14)*20) ms voor de frames
# NB tijd moet nog gecorrigeerd worden voor stimulus length



# declareer alle variabelen




array <int> beginstim1[10];
array <int> beginstim2[10];
array <int> beginstim3[10];
array <int> beginstim4[10];
array <int> beginstim5[10];
array <int> beginstim6[10];
beginstim1.fill(1,10,1,4);
beginstim2.fill(1,10,1,4);
beginstim3.fill(1,10,1,4);
beginstim4.fill(1,10,1,4);
beginstim5.fill(1,10,1,4);
beginstim6.fill(1,10,1,4);

array <int> stimuli[60];
array <int> difficultyF[30];
array <int> difficultyH[30];
# define trial list 
stimuli.fill(1,60,1,1);

difficultyF.fill(1,10,1,0);
difficultyF.fill(11,20,2,0);
difficultyF.fill(21,30,3,0);
difficultyH.fill(1,10,1,0);
difficultyH.fill(11,20,2,0);
difficultyH.fill(21,30,3,0);

# shuffle trial list
stimuli.shuffle();
difficultyF.shuffle();
difficultyH.shuffle();
beginstim1.shuffle();
beginstim2.shuffle();
beginstim3.shuffle();
beginstim4.shuffle();
beginstim5.shuffle();
beginstim6.shuffle();


int stimType=0;
#int newTime=0;
#int bgOr=0;
#int bgDir=0;
#int frameOr=0;
#int frameDir=0;
#int stackOr=0;
#int stackDir=0;
#int portStim= 0;
int cDiffH=1;
int cDiffF=1;
int Difflevel=0;
int cDiff1F=1;
int cDiff2F=1;
int cDiff3F=1;
int cDiff1H=1;
int cDiff2H=1;
int cDiff3H=1;
int correct1=0;
int correct2=0;
int correct3=0;
int ac1;
int ac2;
int ac3;



	
loop
	int cTrial=1
until
	cTrial>60
	
	
begin



	# bepaal type stimulus en pulstijd
	
	if(stimuli[cTrial]<31)  then
		# frame
		stimType=2;
		Difflevel=difficultyF[cDiffF];
		cDiffF=cDiffF+1;
		
		
		
		#newTime=((stimuli[cTrial]-14)*20);
		
	else
		# stack
		stimType=3;
		Difflevel=difficultyH[cDiffH];
		cDiffH=cDiffH+1;
		#newTime=(stimuli[cTrial]*20);
		
		
	end;


	


	# her-teken de schermen
	
	# achtergronden
	
		fixScreenRed.set_part(1,D45[4]);
		fixScreenGreen.set_part(1,D45[4]);
		motionFrame1.set_part(1,D45[3]);
		motionFrame2.set_part(1,D45[2]);
		motionFrame3.set_part(1,D45[1]);


	# faces

			if stimType==2 && Difflevel == 1 then 
				fixScreenRed.set_part(2,F1_x[beginstim1[cDiff1F]]);
				fixScreenGreen.set_part(2,F1_x[beginstim1[cDiff1F]]);
				motionFrame1.set_part(2,F1_x[beginstim1[cDiff1F]+1]);
				motionFrame2.set_part(2,F1_x[beginstim1[cDiff1F]+2]);
				motionFrame3.set_part(2,F1_x[beginstim1[cDiff1F]+3]);
				out.print(beginstim1[cDiff1F]);
				cDiff1F=cDiff1F+1;
				huisgezichtEvent.set_target_button(2);
				

			end; 

			if stimType==2 && Difflevel == 2 then 
				fixScreenRed.set_part(2,F2_x[beginstim2[cDiff2F]]);
				fixScreenGreen.set_part(2,F2_x[beginstim2[cDiff2F]]);
				motionFrame1.set_part(2,F2_x[beginstim2[cDiff2F]+1]);
				motionFrame2.set_part(2,F2_x[beginstim2[cDiff2F]+2]);
				motionFrame3.set_part(2,F2_x[beginstim2[cDiff2F]+3]);
				out.print(beginstim2[cDiff2F]);
				cDiff2F=cDiff2F+1;
				huisgezichtEvent.set_target_button(2);
				

			end;

			if stimType==2 && Difflevel == 3 then 
				fixScreenRed.set_part(2,F3_x[beginstim3[cDiff3F]]);
				fixScreenGreen.set_part(2,F3_x[beginstim3[cDiff3F]]);
				motionFrame1.set_part(2,F3_x[beginstim3[cDiff3F]+1]);
				motionFrame2.set_part(2,F3_x[beginstim3[cDiff3F]+2]);
				motionFrame3.set_part(2,F3_x[beginstim3[cDiff3F]+3]);
				out.print(beginstim3[cDiff3F]);
				cDiff3F=cDiff3F+1;
				huisgezichtEvent.set_target_button(2);
				

			end;


	#house

			if stimType==3 && Difflevel == 1  then 
				fixScreenRed.set_part(2,H1_x[beginstim4[cDiff1H]]);
				fixScreenGreen.set_part(2,H1_x[beginstim4[cDiff1H]]);
				motionFrame1.set_part(2,H1_x[beginstim4[cDiff1H]+1]);
				motionFrame2.set_part(2,H1_x[beginstim4[cDiff1H]+2]);
				motionFrame3.set_part(2,H1_x[beginstim4[cDiff1H]+3]);
				out.print(beginstim4[cDiff1H]);
				cDiff1H=cDiff1H+1;
				huisgezichtEvent.set_target_button(3);
				
			end; 

			if stimType==3 && Difflevel == 2 then 
				fixScreenRed.set_part(2,H2_x[beginstim5[cDiff2H]]);
				fixScreenGreen.set_part(2,H2_x[beginstim5[cDiff2H]]);
				motionFrame1.set_part(2,H2_x[beginstim5[cDiff2H]+1]);
				motionFrame2.set_part(2,H2_x[beginstim5[cDiff2H]+2]);
				motionFrame3.set_part(2,H2_x[beginstim5[cDiff2H]+3]);
				out.print(beginstim5[cDiff2H]);				
				cDiff2H=cDiff2H+1;
				huisgezichtEvent.set_target_button(3);
				
			end; 

			if stimType==3 && Difflevel == 3 then 
				fixScreenRed.set_part(2,H3_x[beginstim6[cDiff3H]]);
				fixScreenGreen.set_part(2,H3_x[beginstim6[cDiff3H]]);
				motionFrame1.set_part(2,H3_x[beginstim6[cDiff3H]+1]);
				motionFrame2.set_part(2,H3_x[beginstim6[cDiff3H]+2]);
				motionFrame3.set_part(2,H3_x[beginstim6[cDiff3H]+3]);
				out.print(beginstim6[cDiff3H]);
				cDiff3H=cDiff3H+1;
				huisgezichtEvent.set_target_button(3);
			end; 

		

	
	# nu nog de goede target-button instellen
	
	
	#zekerEvent.set_target_button(stimType);
	#motion1Event.set_port_code(portStim);
	mainTrial.present();
	
	 
	huisgezichtTrial.present();
	stimulus_data last_rt = stimulus_manager.last_stimulus_data(); 
	int rt_stim = last_rt.reaction_time();
	int res_huisgezicht = response_manager.last_response();	
	if res_huisgezicht == 3 then feedbackstimEvent.set_stimulus(huisresponse);end; 
	if res_huisgezicht == 2 then feedbackstimEvent.set_stimulus(gezichtresponse);end;

	if stimType == res_huisgezicht && Difflevel ==1 then correct1=correct1+1; end;
	if stimType == res_huisgezicht && Difflevel ==2 then correct2=correct2+1; end;
	if stimType == res_huisgezicht && Difflevel ==3 then correct3=correct3+1; end;
	
	
	feedbackstimTrial.present();
	
	zekerTrial.present();
	stimulus_data last_RT = stimulus_manager.last_stimulus_data(); 
	int rt_zeker = last_RT.reaction_time();
	int res_zeker = response_manager.last_response();	
	if res_zeker == 1 then feedbackzekerEvent.set_stimulus(antwoord1);end; 
	if res_zeker == 2 then feedbackzekerEvent.set_stimulus(antwoord2);end;	
	if res_zeker == 3 then feedbackzekerEvent.set_stimulus(antwoord3);end;
	if res_zeker == 4 then feedbackzekerEvent.set_stimulus(antwoord4);end;

		
	feedbackzekerTrial.present();

	# schrijf de output naar een file
	out.print("\t");
	out.print(stimType);
	out.print("\t");
	
	out.print(Difflevel);
	out.print("\t");		
	
	out.print(res_huisgezicht);
	out.print("\t");	
		
	out.print(res_zeker);
	out.print("\t");	
	
	out.print(correct1);
	out.print("\t");
	
	out.print(correct2);
	out.print("\t");	
	
	out.print(correct3);
	out.print("\t");	
	
	out.print(rt_stim);
	out.print("\t");

	out.print(rt_zeker);
	out.print("\t");	
	out.print("\n");
	
	cTrial=cTrial+1;
	
end;
endoftask.present();