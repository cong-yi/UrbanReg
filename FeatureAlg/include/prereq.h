#if defined _WIN32 || defined __CYGWIN__
  #ifdef FEATUREALG_EXPORT
    #ifdef __GNUC__
      #define FEATUREALG_PUBLIC __attribute__ ((dllexport))
    #else
      #define FEATUREALG_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define FEATUREALG_PUBLIC __attribute__ ((dllimport))
    #else
      #define FEATUREALG_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define FEATUREALG_LOCAL
#else
  #if __GNUC__ >= 4
    #define FEATUREALG_PUBLIC __attribute__ ((visibility ("default")))
    #define FEATUREALG_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define FEATUREALG_PUBLIC
    #define FEATUREALG_LOCAL
  #endif
#endif