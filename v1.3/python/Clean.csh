#!/bin/csh -f

/bin/rm -rf __pycache__

foreach F ( * )
    if ( -d $F ) then
    if ( -e $F/Clean.csh ) then
       ( cd $F; ./Clean.csh )
    endif
    endif
end
