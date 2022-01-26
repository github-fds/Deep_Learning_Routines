#!/bin/csh -f

/bin/rm -rf lib
/bin/rm -rf include

foreach F ( * )
    if ( -d $F ) then
    if ( -e $F/Clean.csh ) then
       ( cd $F; ./Clean.csh )
    endif
    endif
end
