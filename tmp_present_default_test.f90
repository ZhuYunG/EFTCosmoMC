module m
contains
  function PresentDefault_S(default, S) result(Sout)
    character(len=*),intent(in),target :: default
    character(len=*),intent(in),target,optional :: S
    character(len=:),pointer :: Sout
    if (present(S)) then
      Sout => S
    else
      Sout => default
    endif
  end function PresentDefault_S
end module m

program t
  use m
  call test
contains
  subroutine test()
    write(*,PresentDefault_S('(a)')) 'hello world'
  end subroutine
end program t
