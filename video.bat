::::del Error/error.csv
::
::
::
::::
::del Velocity_aprox.mp4
::del Pressure_aprox.mp4
::del Ufield_aprox.mp4
::del Vfield_aprox.mp4
::
::set "target_folder=Frames"
::
::echo Deleting all files in subfolders of %target_folder%...
::
::rem 
::cd /d "%target_folder%"
::rem 
::for /r %%i in (*) do del "%%i"
::cd ..
::::::
::::::
::
::::
python src/source/main_kovasznay.py 16
python src/source/main_kovasznay.py 32
python src/source/main_kovasznay.py 64
python src/source/main_kovasznay.py 128

::Vector field video
::ffmpeg -hide_banner -loglevel error -framerate 60 -i Frames\VectorFrames\InterpolatedExact_%%d.png -c:v libx264 -pix_fmt yuv420p Velocity_exact.mp4
::ffmpeg -hide_banner -loglevel error -framerate 60 -i Frames\VectorFrames\VectorFrame_%%d.png -vf scale=1200:400 -c:v libx264 -pix_fmt yuv420p Velocity_aprox.mp4
::ffmpeg -hide_banner -loglevel error -framerate 60 -i Frames\ScalarFrames\Pressure\Pressure_%%d.png -c:v libx264 -pix_fmt yuv420p Pressure_aprox.mp4
::ffmpeg -hide_banner -loglevel error -framerate 60 -i Frames\ScalarFrames\U\UField_%%d.png -c:v libx264 -pix_fmt yuv420p Ufield_aprox.mp4
::ffmpeg -hide_banner -loglevel error -framerate 60 -i Frames\ScalarFrames\V\VField_%%d.png -c:v libx264 -pix_fmt yuv420p Vfield_aprox.mp4
:::ffmpeg -hide_banner -loglevel error -framerate 60 -i Frames\ScalarFrames\Pressure_Exact_%%d.png -c:v libx264 -pix_fmt yuv420p Pressure_exact.mp4
::::python bin/error_plotter.py
