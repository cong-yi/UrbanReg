#if defined _WIN32 || defined __CYGWIN__
  #ifdef FGR_EXPORT
    #ifdef __GNUC__
      #define FGR_PUBLIC __attribute__ ((dllexport))
    #else
      #define FGR_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define FGR_PUBLIC __attribute__ ((dllimport))
    #else
      #define FGR_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define FGR_LOCAL
#else
  #if __GNUC__ >= 4
    #define FGR_PUBLIC __attribute__ ((visibility ("default")))
    #define FGR_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define FGR_PUBLIC
    #define FGR_LOCAL
  #endif
#endif