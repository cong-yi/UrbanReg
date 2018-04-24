#if defined _WIN32 || defined __CYGWIN__
  #ifdef OVERLAPTRIMMER_EXPORT
    #ifdef __GNUC__
      #define OVERLAPTRIMMER_PUBLIC __attribute__ ((dllexport))
    #else
      #define OVERLAPTRIMMER_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define OVERLAPTRIMMER_PUBLIC __attribute__ ((dllimport))
    #else
      #define OVERLAPTRIMMER_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define OVERLAPTRIMMER_LOCAL
#else
  #if __GNUC__ >= 4
    #define OVERLAPTRIMMER_PUBLIC __attribute__ ((visibility ("default")))
    #define OVERLAPTRIMMER_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define OVERLAPTRIMMER_PUBLIC
    #define OVERLAPTRIMMER_LOCAL
  #endif
#endif