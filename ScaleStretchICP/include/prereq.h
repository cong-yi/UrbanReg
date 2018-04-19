#if defined _WIN32 || defined __CYGWIN__
  #ifdef SSICP_EXPORT
    #ifdef __GNUC__
      #define SSICP_PUBLIC __attribute__ ((dllexport))
    #else
      #define SSICP_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define SSICP_PUBLIC __attribute__ ((dllimport))
    #else
      #define SSICP_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define SSICP_LOCAL
#else
  #if __GNUC__ >= 4
    #define SSICP_PUBLIC __attribute__ ((visibility ("default")))
    #define SSICP_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define SSICP_PUBLIC
    #define SSICP_LOCAL
  #endif
#endif