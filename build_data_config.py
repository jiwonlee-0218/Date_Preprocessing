def run(data_settings_yml):
    """Generate and write out a CPAC data configuration (participant list)
    YAML file."""

    import os
    import yaml
    import CPAC

    print("\nGenerating data configuration file..")

    settings_dct = yaml.safe_load(open(data_settings_yml, 'r'))

    if "awsCredentialsFile" not in settings_dct or \
            not settings_dct["awsCredentialsFile"]:
        settings_dct["awsCredentialsFile"] = None
    elif "None" in settings_dct["awsCredentialsFile"] or \
            "none" in settings_dct["awsCredentialsFile"]:
        settings_dct["awsCredentialsFile"] = None

    if "anatomical_scan" not in settings_dct or \
        not settings_dct["anatomical_scan"]:
        settings_dct["anatomical_scan"] = None
    elif "None" in settings_dct["anatomical_scan"] or \
            "none" in settings_dct["anatomical_scan"]:
        settings_dct["anatomical_scan"] = None

    # inclusion lists
    incl_dct = format_incl_excl_dct(settings_dct.get('siteList', None), 'sites')
    incl_dct.update(format_incl_excl_dct(settings_dct.get('subjectList', None),
                                         'participants'))
    incl_dct.update(format_incl_excl_dct(settings_dct.get('sessionList', None),
                                         'sessions'))
    incl_dct.update(format_incl_excl_dct(settings_dct.get('scanList', None), 'scans'))

    # exclusion lists
    excl_dct = format_incl_excl_dct(settings_dct.get('exclusionSiteList', None),
                                    'sites')
    excl_dct.update(format_incl_excl_dct(settings_dct.get('exclusionSubjectList', None),
                                         'participants'))
    excl_dct.update(format_incl_excl_dct(settings_dct.get('exclusionSessionList', None),
                                         'sessions'))
    excl_dct.update(format_incl_excl_dct(settings_dct.get('exclusionScanList', None),
                                         'scans'))

    if 'bids' in settings_dct['dataFormat'].lower():

        file_list = get_file_list(settings_dct["bidsBaseDir"],
                                  creds_path=settings_dct["awsCredentialsFile"])

        data_dct = get_BIDS_data_dct(settings_dct['bidsBaseDir'],
                                     file_list=file_list,
                                     brain_mask_template=settings_dct['brain_mask_template'],
                                     anat_scan=settings_dct['anatomical_scan'],
                                     aws_creds_path=settings_dct['awsCredentialsFile'],
                                     inclusion_dct=incl_dct,
                                     exclusion_dct=excl_dct,
                                     config_dir=settings_dct["outputSubjectListLocation"])

    elif 'custom' in settings_dct['dataFormat'].lower():

        # keep as None if local data set (not on AWS S3 bucket)
        file_list = None
        base_dir = None

        if "s3://" in settings_dct["anatomicalTemplate"]:
            # hosted on AWS S3 bucket
            if '{site}' in settings_dct["anatomicalTemplate"]:
                base_dir = \
                    settings_dct["anatomicalTemplate"].split('{site}')[0]
            elif '{participant}' in settings_dct["anatomicalTemplate"]:
                base_dir = \
                    settings_dct["anatomicalTemplate"].split('{participant}')[0]

        elif "s3://" in settings_dct["functionalTemplate"]:
            # hosted on AWS S3 bucket
            if '{site}' in settings_dct["functionalTemplate"]:
                base_dir = \
                    settings_dct["functionalTemplate"].split('{site}')[0]
            elif '{participant}' in settings_dct["functionalTemplate"]:
                base_dir = \
                    settings_dct["functionalTemplate"].split('{participant}')[0]

        if base_dir:
            file_list = pull_s3_sublist(base_dir,
                                        settings_dct['awsCredentialsFile'])

        params_dct = None
        if settings_dct['scanParametersCSV']:
            if '.csv' in settings_dct['scanParametersCSV']:
                params_dct = \
                    extract_scan_params_csv(settings_dct['scanParametersCSV'])

        data_dct = get_nonBIDS_data(settings_dct['anatomicalTemplate'],
                                    settings_dct['functionalTemplate'],
                                    file_list=file_list,
                                    anat_scan=settings_dct['anatomical_scan'],
                                    scan_params_dct=params_dct,
                                    brain_mask_template=settings_dct['brain_mask_template'],
                                    fmap_phase_template=settings_dct['fieldMapPhase'],
                                    fmap_mag_template=settings_dct['fieldMapMagnitude'],
                                    aws_creds_path=settings_dct['awsCredentialsFile'],
                                    inclusion_dct=incl_dct,
                                    exclusion_dct=excl_dct)

    else:
        err = "\n\n[!] You must select a data format- either 'BIDS' or " \
              "'Custom', in the 'dataFormat' field in the data settings " \
              "YAML file.\n\n"
        raise Exception(err)

    if len(data_dct) > 0:



        data_config_outfile = \
            os.path.join(settings_dct['outputSubjectListLocation'],
                         "data_config_{0}.yml"
                         "".format(settings_dct['subjectListName']))

        group_list_outfile = \
            os.path.join(settings_dct['outputSubjectListLocation'],
                         "group_analysis_participants_{0}.txt"
                         "".format(settings_dct['subjectListName']))

        # put data_dct contents in an ordered list for the YAML dump
        data_list = []
        group_list = []

        included = {'site': [], 'sub': []}
        num_sess = num_scan = 0

        for site in sorted(data_dct.keys()):
            for sub in sorted(data_dct[site]):
                for ses in sorted(data_dct[site][sub]):
                    # if there are scans, get some numbers
                    included['site'].append(site)
                    included['sub'].append(sub)
                    num_sess += 1
                    if 'func' in data_dct[site][sub][ses]:
                        for scan in data_dct[site][sub][ses]['func']:
                            num_scan += 1

                    data_list.append(data_dct[site][sub][ses])
                    group_list.append("{0}_{1}".format(sub, ses))

        # calculate numbers
        num_sites = len(set(included['site']))
        num_subs = len(set(included['sub']))

        with open(data_config_outfile, "wt") as f:
            # Make sure YAML doesn't dump aliases (so it's more human
            # read-able)
            f.write("# CPAC Data Configuration File\n# Version {0}"
                    "\n".format(CPAC.__version__))
            f.write("#\n# http://fcp-indi.github.io for more info.\n#\n"
                    "# Tip: This file can be edited manually with "
                    "a text editor for quick modifications.\n\n")
            noalias_dumper = yaml.dumper.SafeDumper
            noalias_dumper.ignore_aliases = lambda self, data: True
            f.write(yaml.dump(data_list, default_flow_style=False,
                              Dumper=noalias_dumper))

        with open(group_list_outfile, "wt") as f:
            # write the inclusion list (mainly the group analysis sublist)
            # text file
            for id in sorted(group_list):
                f.write("{0}\n".format(id))

        if os.path.exists(data_config_outfile):
            print("\nCPAC DATA SETTINGS file entered (use this preset file " \
                  "to modify/regenerate the data configuration file):" \
                  "\n{0}\n".format(data_settings_yml))
            print("Number of:")
            print("...sites: {0}".format(num_sites))
            print("...participants: {0}".format(num_subs))
            print("...participant-sessions: {0}".format(num_sess))
            print("...functional scans: {0}".format(num_scan))
            print("\nCPAC DATA CONFIGURATION file created (use this for " \
                  "individual-level analysis):" \
                  "\n{0}\n".format(data_config_outfile))

        if os.path.exists(group_list_outfile):
            print("Group-level analysis participant-session list text " \
                  "file created (use this for group-level analysis):\n{0}" \
                  "\n".format(group_list_outfile))

    else:
        err = "\n\n[!] No anatomical input files were found given the data " \
              "settings provided.\n\n"
        raise Exception(err)



run('/home/djk/abide_data_setting.yml')