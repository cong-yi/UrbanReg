#if defined _WIN32 || defined __CYGWIN__
  #ifdef BASEALG_EXPORT
    #ifdef __GNUC__
      #define BASEALG_PUBLIC __attribute__ ((dllexport))
    #else
      #define BASEALG_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define BASEALG_PUBLIC __attribute__ ((dllimport))
    #else
      #define BASEALG_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define BASEALG_LOCAL
#else
  #if __GNUC__ >= 4
    #define BASEALG_PUBLIC __attribute__ ((visibility ("default")))
    #define BASEALG_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define BASEALG_PUBLIC
    #define BASEALG_LOCAL
  #endif
#endif